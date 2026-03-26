import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

from quantize.triton_utils.kernels import dequant_dim0, dequant_dim1
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from tqdm import tqdm
import gc  
from quantize.utils import get_named_linears,set_op_by_name

logger = getLogger(__name__)


class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class QuantLinear(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        **kwargs
    ):
        super().__init__()
        # if bits not in [2, 4, 8]:
        #     raise NotImplementedError("Only 2,4,8 bits are supported.")
        # if infeatures % 32 != 0 or outfeatures % 32 != 0:
        #     raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = math.ceil(bits)
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.register_buffer(
            'qweight',
            torch.zeros((infeatures, outfeatures), dtype=torch.uint8)
        )
        self.register_parameter(
            'scales',
            torch.nn.Parameter(torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16))
        )
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16)
        )

        self.register_buffer(
            'col_order',
            torch.zeros(infeatures, dtype=torch.int32)
        )
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.zeros_dim0, self.zeros_dim1 = self.scales.shape
        self.trainable = trainable
        self.scales.requires_grad = True
        self.use_fake = False

    def post_init(self):
        pass


    def use_fake_quantization(self, del_quant=False,transpose=False):
        # use fake quantization for faster training but consume more memory
        weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
        dim0, dim1 = weight.shape
        zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
        weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        if transpose:
            self.fake_transpose = True
            weight = weight.transpose(0,1).contiguous()
        self.register_buffer(
            'weight',
            weight
        )
        self.use_fake = True
        if del_quant:
            del self.qweight
            del self.scales
            del self.qzeros
            del self.g_idx
        
    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()
        
        self.scales = nn.Parameter(scales.half())
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        self.col_order = linear.weight_quantizer.col_order


        self.qweight = linear.get_intweight().t().contiguous()

        self.qzeros = zeros.to(torch.float16)

    def forward(self, x):
        if self.use_fake:
            weight = self.weight
            if self.fake_transpose:
                weight = weight.transpose(0,1)
        else:
            # weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)

            weight = self.qweight.to(torch.float16)

            invcol = torch.argsort(self.col_order)

            weight = weight[self.col_order, : ]
            dim0, dim1 = weight.shape
            # dim2 = (dim1*dim0)//self.group_size
            zeros = self.qzeros.to(torch.float16)
            weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)

            weight = weight[invcol, : ]
        # out = torch.matmul(x, weight)
        out = torch.matmul(x, weight.to(x.dtype))
        out = out + self.bias if self.bias is not None else out
        return out


def load_quantized_model(model_path, wbits, group_size):
    print(f"Loading quantized model from {model_path}")

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=config.dtype, trust_remote_code=True)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")

    return model, tokenizer

def load_quantized_model_multi_devices(model_path, wbits, group_size, model_type:str="llama") :
    print(f"Loading quantized model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_path)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=config.dtype, trust_remote_code=True)

    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Replace linears with QuantLinear"):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(wbits, group_size, module.in_features, module.out_features, module.bias is not None)
            set_op_by_name(layer, name, q_linear)

    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    block_class_name = model.model.layers[0].__class__.__name__
    max_memory = {0:"2GB",1:"10GB",2:"10GB"}
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=[block_class_name])
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    model.gradient_checkpointing_enable()


    return model, tokenizer



__all__ = ["QuantLinear","load_omniq_quantized"]
