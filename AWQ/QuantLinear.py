import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import math
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from utils import get_named_linears,set_op_by_name

import gc

class QuantLinear(nn.Module):

    def __init__(
        self,
        bits,
        group_size,
        outfeature_interval,
        infeatures,
        outfeatures,
        bias,
        **kwargs
    ):
        super().__init__()
        assert bits <= 4.0 and type(bits) == float, "Only support bits <= 4 and float type"
        assert infeatures % group_size == 0 or group_size == -1, "infeatures must be divisible by group_size"
        assert outfeatures % outfeature_interval == 0, "outfeatures must be divisible by outfeature_interval"

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.outfeature_interval = outfeature_interval  # row_interal
        self.bits = bits
        self.low_bit  = math.floor(bits)  # b-
        self.high_bit = math.ceil(bits)   # b+
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.high_bit - 1

        self.register_buffer(
            'qweight_metadata_len', 
            torch.tensor(0)
        ) 

        self.register_buffer(
            'in_reorder',
            torch.zeros((infeatures,), dtype=torch.int32)
        )

        self.register_buffer(
            'out_reorder',
            torch.zeros((outfeatures,), dtype=torch.int32)
        )

        self.register_buffer(  
            'intweight',
            torch.zeros((infeatures,outfeatures), dtype=torch.int8)
        )

        self.register_buffer(
            'scales',
            torch.nn.Parameter(torch.zeros((infeatures // self.group_size, outfeatures), dtype=torch.float16))
        )
        self.register_buffer(
            'zeros',
            torch.zeros((infeatures // self.group_size, outfeatures), dtype=torch.float16)
        )

        self.register_buffer(
            'block_bitwidth',
            torch.zeros((infeatures // self.group_size, outfeatures // outfeature_interval), dtype=torch.int8)
        )

        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        
    def pack(self, in_reorder, out_reorder, intweight, scales, zeros, block_bitwidth):

        self.in_reorder = in_reorder.reshape(in_reorder.shape).to(torch.int32)
        self.out_reorder = out_reorder.reshape(out_reorder.shape).to(torch.int32)
        self.intweight = intweight.to(torch.int8)
        self.scales = scales.half()
        self.zeros = zeros.half()
        self.block_bitwidth = block_bitwidth.to(torch.int8)

        qweight_metadata_len = block_bitwidth.to(torch.int32).sum().item() *self.outfeature_interval * self.group_size // 32
        self.qweight_metadata_len = torch.tensor(qweight_metadata_len)
    def forward(self, x):

        x = x[:, :, self.in_reorder]
        dim0, dim1 = self.intweight.shape
        weight = self.intweight.to(x.dtype)
        weight = ((weight.reshape(-1, self.group_size, dim1) - self.zeros.reshape(-1, 1, dim1)) * self.scales.reshape(-1, 1, dim1)).reshape(dim0, dim1)
        out = torch.matmul(x, weight.to(x.dtype))
        out = out[:, :, self.out_reorder]
        out = out + self.bias if self.bias is not None else out
        return out


def load_quantized_model(model_path, wbits, group_size, outfeature_interval):
    print(f"Loading quantized model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            
            if type(wbits) == dict :
                key = f'model.layers.{i}.{name}'
                n_bits = wbits[key]
            else :
                n_bits = wbits
            q_linear = QuantLinear(n_bits, group_size, outfeature_interval, module.in_features,module.out_features,not module.bias is None)
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