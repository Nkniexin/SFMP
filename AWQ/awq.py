import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RMSNorm
from transformers.activations import GELUActivation
import numpy as np
import json
import os
import time

from base import BASE, get_awq_calib_dataset
from skip_llama import LlamaDecoderSkipLayer
from accelerate import dispatch_model

import gc
from tqdm import tqdm
import math
import functools
from copy import deepcopy

from collections import defaultdict
from sfmp import QuantLinear
from sfmp.bitallocation import bit_allocation, calculate_average_bit, check_mixed_precison_config

mixed_precision_config = {
    "bits":None,
    "quantile":None,
}

def assign_importance_levels(importance, bit :float = 3.5):

    flat = importance.flatten()
    bits = mixed_precision_config["bits"]
    quantile = mixed_precision_config["quantile"]

    num_bits = len(bits)
    num_quantiles = len(quantile)

    levels = torch.full_like(importance, fill_value=bits[0])

    for i in range(num_quantiles) :
        i_quantile = quantile[i]
        levels[importance > i_quantile] = bits[i+1]
    
    return levels 

def direct_block(w,block_h:int = 32,block_w :int = 128,bit : float = 3.5):

    h_blocks = w.shape[0] // block_h
    w_blocks = w.shape[1] // block_w

    importance = w.reshape(h_blocks, block_h, w_blocks, block_w).sum(dim=(1, 3)).to(torch.float32)

    bit_alloc_block = assign_importance_levels(importance,bit)

    bit_alloc_full = bit_alloc_block.repeat_interleave(block_h, dim=0).to(torch.int)

    return bit_alloc_full 

def bit_alloc_block_wise(w,block_h:int = 32,block_w :int = 128,bit : float = 3.5):

    h_blocks = w.shape[0] // block_h
    w_blocks = w.shape[1] // block_w

    importance = w.reshape(h_blocks, block_h, w_blocks, block_w).sum(dim=(1, 3)).to(torch.float32)

    block_width = assign_importance_levels(importance, bit )

    return block_width

@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


def real_quantize_tensor(w, n_bit = None, q_group_size=-1,use_colreorder = False,use_rowreorder = False, 
                         sensitivity_path = None,linear_name = None,layer_idx=None, row_interval = None) :
    
    assert q_group_size != 0, "q_group_size should not be 0"
    assert n_bit is not None, "n_bit should not be None"

    sensitivity_dtype = torch.float32

    if q_group_size == -1:
        q_group_size = w.shape[-1]
    sensitivity = np.load(f'{sensitivity_path}/model.layers.{layer_idx}.{linear_name}.npy')
    sensitivity = torch.tensor(sensitivity,dtype = sensitivity_dtype)

    org_w_shape = w.shape
    dim1,dim2 = sensitivity.shape
    w = w.reshape(dim1,dim2)

    if use_colreorder :
        col_sums = sensitivity.sum(dim=0)            
        col_order = torch.argsort(col_sums, descending=True)  # maybe use descending=False 
        sensitivity = sensitivity[:, col_order] 
        w = w[:, col_order]
        invcol = torch.argsort(col_order)
    else :
        col_order = torch.arange(dim2)

    if use_rowreorder :
        row_sums = sensitivity.sum(dim=1)            
        row_order = torch.argsort(row_sums, descending=True)  # maybe use descending=False 
        sensitivity = sensitivity[row_order, :]   
        w = w[row_order, :]
        invrow = torch.argsort(row_order)
    else :
        row_order = torch.arange(dim1)
        invrow = torch.argsort(row_order)

    score = direct_block(sensitivity, row_interval, q_group_size, bit = n_bit)
    score = score.to(w.device)

    
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
        score = score.reshape(-1, 1)
    elif q_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**score - 1
    min_int = torch.zeros_like(max_int)
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    intweight = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)

    intweight = intweight.reshape(dim1,dim2)

    scales = scales.reshape(dim1,-1)

    zeros = zeros.reshape(dim1,-1)

    block_width = bit_alloc_block_wise(sensitivity,row_interval,q_group_size,n_bit)

    intweight = intweight.t()

    scales = scales.t()

    zeros = zeros.t()

    block_width = block_width.t()

    return intweight, scales, zeros, col_order, invrow, block_width

def pseudo_quantize_tensor(w, n_bit = None, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False,use_colreorder = False,  
                           use_rowreorder = False, sensitivity_path = None,linear_name = None,layer_idx=None, row_interval = None):
    assert q_group_size != 0, "q_group_size should not be 0"
    assert n_bit is not None, "n_bit should not be None"

    sensitivity_dtype = torch.float32
    if q_group_size == -1:
        q_group_size = w.shape[-1]
    sensitivity = np.load(f'{sensitivity_path}/model.layers.{layer_idx}.{linear_name}.npy')
    sensitivity = torch.tensor(sensitivity,dtype = sensitivity_dtype)

    org_w_shape = w.shape
    dim1,dim2 = sensitivity.shape
    w = w.reshape(dim1,dim2)

    if use_colreorder :
        col_sums = sensitivity.sum(dim=0)            
        col_order = torch.argsort(col_sums, descending=True)   
        sensitivity = sensitivity[:, col_order] 
        w = w[:, col_order]
        invcol = torch.argsort(col_order)

    if use_rowreorder :
        row_sums = sensitivity.sum(dim=1)            
        row_order = torch.argsort(row_sums, descending=True)  
        sensitivity = sensitivity[row_order, :]   
        w = w[row_order, :]
        invrow = torch.argsort(row_order)

    score = direct_block(sensitivity, row_interval, q_group_size, bit = n_bit)
    score = score.to(w.device)

    
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
        score = score.reshape(-1, 1)
    elif q_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**score - 1
        min_int = torch.zeros_like(max_int)
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(dim1,dim2)
    if use_rowreorder :
        w = w[invrow, :]
    if use_colreorder :
        w = w[:, invcol]

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
    

def clip_asym_pseudo_quantize_tensor(w, n_bit:float=2.75, zero_point=True, q_group_size=-1, 
                                     inplace=False, get_scale_zp=False,in_channel = None,score= None):
    assert q_group_size != 0, "q_group_size should not be 0"

    if q_group_size == -1:
        q_group_size = w.shape[-1]

    org_w_shape = w.shape

    w = w.reshape(-1,in_channel)

    
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
        score = score.reshape(-1, 1)
    elif q_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**score - 1
        min_int = torch.zeros_like(max_int)
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(-1,in_channel)

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


class AWQ(BASE):
    def __init__(self, model_name, config, arch, device_map, dtype=torch.float16, group_size=128, dev='cuda', prune=False, owq=None, **kwargs):
        super().__init__(model_name, config, arch, device_map=device_map,dtype=dtype, group_size=group_size, dev=dev, prune=prune)
        self.method = 'awq'
        self.sensitivity_path = kwargs.get('sensitivity_path', None)
        
        self.wbits = kwargs.get('wbits', None)
        assert self.wbits is not None,'wbits must be specified'

        self.clip_asym = kwargs.get('clip_asym', True)

        self.use_rowreorder = kwargs.get('use_rowreorder', False)
        print(f'whether use rowreorder : {self.use_rowreorder}')

        self.use_colreorder = kwargs.get('use_colreorder', False)
        print(f'whether use colreorder : {self.use_colreorder}')

        self.row_interval = kwargs.get('row_interval', 32)
        print(f'row_interval : {self.row_interval}')

        if self.clip_asym:
            print("Clipping asymmetrically")
        else:
            print("Clipping symmetrically")


    @torch.no_grad()
    def run_awq(
        self,
        samples=None,
        n_samples=512,
        seqlen=512,
    ):

        if samples is None:
            samples = get_awq_calib_dataset(tokenizer=self.tokenizer, n_samples=n_samples, block_size=seqlen)
            samples = torch.cat(samples, dim=0)

            print(samples.shape)


        layers = self.model.model.layers

        inps = []
        layer_kwargs = {}

        layers[0] = layers[0].to(self.dev)
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.dev)
        self.model.model.norm = self.model.model.norm.to(self.dev)
        self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.dev)


        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                inps.append(inp)
                layer_kwargs.update(kwargs)
                raise ValueError
            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)

        layers[0] = Catcher(layers[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        del samples
        layers[0] = layers[0].module  # restore
        inps = inps[0]

        layers[0] = layers[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.to('cpu')
        self.model.model.norm = self.model.model.norm.to('cpu')
        self.model.model.rotary_emb = self.model.model.rotary_emb.to('cpu')

        gc.collect()
        torch.cuda.empty_cache()

        awq_results = {
            "scale": [],
            "clip": [],
        }

        # solve layer by layer
        for i in tqdm(range(len(layers)), desc="Running AWQ..."):
            layer = layers[i]
            layer = layer.to(self.dev)
            named_linears = self.get_named_linears(layer)

            # firstly, get input features of all linear layers
            def cache_input_hook(m, x, y, name, feat_dict):
                x = x[0]
                x = x.detach().cpu()
                feat_dict[name].append(x)

            input_feat = defaultdict(list)
            handles = []
            for name in named_linears:
                handles.append(
                    named_linears[name].register_forward_hook(
                        functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                    )
                )
            if sum(1 for _ in layer.parameters()):
                inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
            # get output as next layer's input
            inps = layer(inps, **layer_kwargs)
            for h in handles:
                h.remove()
            # now solve for scaling and clipping
            input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

            # Clear GPU memory
            torch.cuda.empty_cache()
            scales_list = self.auto_scale_block_bit_adjust_per_linear_owq(
                layer,
                layer_kwargs,
                input_feat=input_feat,
                module_bit=self.wbits,
                layer_idx = i
            )
            self.apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            awq_results["scale"] += self.append_str_prefix(
                scales_list, self.get_op_name(self.model, layer) + "."
            )

            # Clear GPU memory
            torch.cuda.empty_cache()

            if self.clip_asym:
                clip_list = self.auto_clip_block_asym(
                    layer,
                    input_feat=input_feat,
                    module_bit=self.wbits,
                    layer_idx=i,
                )
                self.apply_clip_asym(layer, clip_list,layer_idx=i)
            else:
                clip_list = self.auto_clip_block_sym(
                    layer,
                    input_feat=input_feat,
                    module_bit = self.wbits,
                    layer_idx=i
                )
                self.apply_clip_sym(layer, clip_list,layer_idx=i)
            clip_list = self.append_str_prefix(
                clip_list, self.get_op_name(self.model, layer) + "."
            )
            awq_results["clip"] += clip_list                

            # Haotian: check activation replacement
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()
        
            layer = layer.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        return awq_results

    @torch.no_grad()
    def auto_scale_block_bit_adjust_per_linear_owq(self, module, module_kwargs, input_feat, module_bit=None,layer_idx=None):

        def w_quantize_func(p, bit=None,linear_name=None):
            return pseudo_quantize_tensor(
                p,
                n_bit=bit,
                q_group_size=self.group_size,
                use_colreorder=self.use_colreorder,
                use_rowreorder=self.use_rowreorder,
                sensitivity_path=self.sensitivity_path,
                linear_name=linear_name,
                layer_idx = layer_idx,
                row_interval=self.row_interval
            ).detach()

        if "use_cache" in module_kwargs:
            module_kwargs.pop("use_cache")

        def _search_module_scale_per_linear(block, linears2scale: dict, x, kwargs={}, module_bit = None,layer_idx=0):
            # w: co, ci
            # x: n, ci
            assert module_bit is not None
            assert isinstance(linears2scale, dict)

            x = x.to(next(block.parameters()).device)
            with torch.no_grad():
                org_out = block(x, **kwargs)
                if isinstance(org_out, tuple):
                    org_out = org_out[0]

            x_max = get_act_scale(x)

            best_error = float("inf")
            best_ratio = -1
            best_scales = None

            n_grid = 20  #n_grid = 20
            history = []

            org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
            for ratio in range(n_grid):
                ratio = ratio * 1 / n_grid
                scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                for fc_name, fc in linears2scale.items():

                    key = f'model.layers.{layer_idx}.{fc_name}'

                    
                    if type(module_bit) is dict:
                        mb = module_bit.get(key, None)
                    else:
                        mb = module_bit

                    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                    fc.weight.data = w_quantize_func(fc.weight.data, bit=mb,linear_name=fc_name) / (scales.view(1, -1))

                out = block(x, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]

                loss = (
                    (org_out - out).float().pow(2).mean().item()
                )  # float prevents overflow
                history.append(loss)
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales
                block.load_state_dict(org_sd)
            if best_ratio == -1:
                print(history)
                raise Exception
            best_scales = best_scales.view(-1)

            assert torch.isnan(best_scales).sum() == 0, best_scales
            return best_scales.detach()


        def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}, module_bit=None,layer_idx=0):
            # module2inspect: if given, we will check the output diff of this module instead of layers
            if module2inspect is None:
                assert len(layers) == 1
                module2inspect = list(layers.values())[0]

            scales = _search_module_scale_per_linear(module2inspect, layers, inp, kwargs, module_bit=module_bit,layer_idx=layer_idx)
            scales = scales.detach().cpu()
            # prev_op_name, [layer_name], scale
            return (
                self.get_op_name(module, prev_op),
                tuple([self.get_op_name(module, m) for m in layers.values()]),
                scales,
            )

        scales_list = []  # return the searched scales

        if isinstance(module, LlamaDecoderLayer) or isinstance(module, LlamaDecoderSkipLayer) or isinstance(module, Qwen3DecoderLayer):
            if isinstance(module, (LlamaDecoderLayer, Qwen3DecoderLayer)) or module.attn_skipped is False:
                # attention input
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.input_layernorm, 
                        # layers=[
                        #     module.self_attn.q_proj,
                        #     module.self_attn.k_proj,
                        #     module.self_attn.v_proj,
                        # ],
                        layers={
                            'self_attn.q_proj': module.self_attn.q_proj,
                            'self_attn.k_proj': module.self_attn.k_proj,
                            'self_attn.v_proj': module.self_attn.v_proj,
                        },
                        inp=input_feat["self_attn.q_proj"],
                        module2inspect=module.self_attn,
                        kwargs=module_kwargs,
                        module_bit=module_bit,
                        layer_idx=layer_idx,
                    )
                )
                # attn out
                # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                    scales_list.append(
                        _auto_get_scale(
                            prev_op=module.self_attn.v_proj,
                            # layers=[module.self_attn.o_proj],
                            layers={
                                'self_attn.o_proj': module.self_attn.o_proj,
                            },
                            inp=input_feat["self_attn.o_proj"],
                            module_bit=module_bit,
                            layer_idx=layer_idx,
                        )
                    )
            if isinstance(module, (LlamaDecoderLayer, Qwen3DecoderLayer)) or module.mlp_skipped is False:
                # fc1
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.post_attention_layernorm,
                        # layers=[module.mlp.gate_proj, module.mlp.up_proj],
                        layers={
                            'mlp.gate_proj': module.mlp.gate_proj,
                            'mlp.up_proj': module.mlp.up_proj,
                        },
                        inp=input_feat["mlp.gate_proj"],
                        module2inspect=module.mlp,
                        module_bit=module_bit,
                        layer_idx=layer_idx
                    )
                )
                # fc2
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.mlp.up_proj,
                        # layers=[module.mlp.down_proj],
                        layers={
                            'mlp.down_proj': module.mlp.down_proj,
                        },
                        inp=input_feat["mlp.down_proj"],
                        module_bit=module_bit,
                        layer_idx= layer_idx,
                    )
                )
        else:
            raise NotImplementedError(f"{type(module)} not supported yet!")

        return scales_list
    

    def apply_scale(self, module, scales_list, input_feat_dict=None):
        for prev_op_name, layer_names, scales in scales_list:
            prev_op = self.get_op_by_name(module, prev_op_name)
            layers = [self.get_op_by_name(module, name) for name in layer_names]

            prev_op.cuda()
            for layer in layers:
                layer.cuda()
            scales.cuda()

            if isinstance(prev_op, nn.Linear):
                assert len(layers) == 1
                scale_fc_fc(prev_op, layers[0], scales)
            elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm, Qwen3RMSNorm)):
                scale_ln_fcs(prev_op, layers, scales)
            elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
                new_module = self.ScaledActivation(prev_op, scales)
                self.set_op_by_name(module, prev_op_name, new_module)
                scale_gelu_fc(prev_op, layers[0], scales)
            else:
                raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

            # apply the scaling to input feat if given; prepare it for clipping
            if input_feat_dict is not None:
                for layer_name in layer_names:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))

            prev_op.cpu()
            for layer in layers:
                layer.cpu()
            scales.cpu()


    @torch.no_grad()
    def auto_clip_block_asym(self, module, input_feat, module_bit=None,layer_idx=None):
        named_linears = {
            name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
        }
        clip_list = []
        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                continue      
            named_linears[name].to(self.dev)
            q_config = {}
            key = f'model.layers.{layer_idx}.{name}'
            if type(module_bit) is dict:
                mb = module_bit.get(key, None)
            else:
                mb = module_bit


            q_config['q_group_size'] = self.group_size
            max_val, min_val = self.auto_clip_layer_asym(
                named_linears[name].weight, input_feat[name], n_bit=mb, q_config=q_config,
                layer_idx=layer_idx,linear_name=name
            )
            clip_list.append((name, max_val, min_val))
            
            named_linears[name].cpu()
        return clip_list

    @torch.no_grad()
    def auto_clip_layer_asym(
        self, w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512,linear_name=None,layer_idx=None  # n_grid = 20
    ):  
        assert type(n_bit) == float, "bit should be float"
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = (
            q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
        )

        sensitivity_dtype = torch.float32

        sensitivity = np.load(f'{self.sensitivity_path}/model.layers.{layer_idx}.{linear_name}.npy')
        sensitivity = torch.tensor(sensitivity,dtype=sensitivity_dtype,device=w.device)

        if self.use_colreorder :

            col_sums = sensitivity.sum(dim=0)            
            col_order = torch.argsort(col_sums, descending=True) # maybe use descending=False 
            sensitivity = sensitivity[:, col_order] 
            invcol = torch.argsort(col_order)

        if self.use_rowreorder :
            row_sums = sensitivity.sum(dim=1)            
            row_order = torch.argsort(row_sums, descending=True)  # maybe use descending=False 
            sensitivity = sensitivity[row_order, :]   
            invrow = torch.argsort(row_order)

        score = direct_block(sensitivity, self.row_interval, group_size, bit = n_bit)
        score = score.to(w.device)

        if self.use_rowreorder :
            score = score[invrow, :]


        input_feat = input_feat.view(-1, input_feat.shape[-1])

        if self.use_colreorder :
            input_feat  = input_feat[:, col_order.to(input_feat.device)]    # maybe use col_order

        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
        input_feat = input_feat.to(w.device)

        if self.use_colreorder :
            w = w[:, col_order]    # maybe use col_order
        w = w.reshape(w.shape[0], 1, -1, group_size)


        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        # oc_batch_size = w.shape[0]
        assert w.shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []
        best_min_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group
            
            org_max_val = w.amax(dim=-1, keepdim=True)
            org_min_val = w.amin(dim=-1, keepdim=True)
            assert torch.isinf(org_max_val).sum() == 0, org_max_val
            assert torch.isinf(org_min_val).sum() == 0, org_min_val

            best_max_val = org_max_val.clone()
            best_min_val = org_min_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = org_min_val * (1 - i_s / n_grid)
                cur_w = torch.clamp(w, min_val, max_val)

                q_w = clip_asym_pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config, in_channel = org_w_shape[1] ,score = score[i_b*oc_batch_size:(i_b+1)*oc_batch_size, :])  # co, 1, n_group, group size

                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w, cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
                best_min_val[cur_best_idx] = min_val[cur_best_idx]
            best_max_val_all.append(best_max_val)
            best_min_val_all.append(best_min_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)
        best_min_val = torch.cat(best_min_val_all, dim=0)

        del input_feat, org_out
        gc.collect()
        torch.cuda.empty_cache()
        return best_max_val.squeeze(1), best_min_val.squeeze(1)


    @torch.no_grad()
    def apply_clip_asym(self, module, clip_list,layer_idx=None):
        for name, max_val, min_val in clip_list:

            layer = self.get_op_by_name(module, name)

            layer.to(self.dev)
            max_val = max_val.to(layer.weight.device)
            min_val = min_val.to(layer.weight.device)
            org_shape = layer.weight.shape

            if layer_idx != None :
                sensitivity = np.load(f'{self.sensitivity_path}/model.layers.{layer_idx}.{name}.npy')
            else :
                sensitivity = np.load(f'{self.sensitivity_path}/{name}.npy')

            sensitivity_dtype = torch.float32
            sensitivity = torch.tensor(sensitivity,dtype=sensitivity_dtype,device=layer.weight.device)
            if self.use_colreorder :
                col_sums = sensitivity.sum(dim=0)            
                col_order = torch.argsort(col_sums, descending=True)  
                sensitivity = sensitivity[:, col_order] 
                invcol = torch.argsort(col_order)
                layer.weight.data = layer.weight.data[:,col_order]


            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)            
            layer.weight.data = torch.clamp(layer.weight.data, min_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)

            if self.use_colreorder :
                layer.weight.data = layer.weight.data[:, invcol]

            torch.cuda.empty_cache()
            gc.collect()
            layer.cpu()

    @torch.no_grad()
    def auto_clip_block_sym(self, module, input_feat, module_bit = None,layer_idx=None):
        named_linears = {
            name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
        }
        clip_list = []
        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                continue
            named_linears[name].cuda()
            q_config = {}

            key = f'model.layers.{layer_idx}.{name}'
            if type(module_bit) is dict:
                mb = module_bit.get(key, None)
            else:
                mb = module_bit

            q_config['q_group_size'] = self.group_size

            max_val = self.auto_clip_layer_sym(
                named_linears[name].weight, input_feat[name], n_bit=mb, q_config=q_config,
                layer_idx=layer_idx,linear_name=name
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()
        return clip_list



    @torch.no_grad()
    def auto_clip_layer_sym(
        self, w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512,linear_name=None,layer_idx=None
    ):
        
        assert type(n_bit) == float, "bit should be float"
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = (
            q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
        )

        sensitivity_dtype = torch.float32

        sensitivity = np.load(f'{self.sensitivity_path}/model.layers.{layer_idx}.{linear_name}.npy')
        sensitivity = torch.tensor(sensitivity,dtype=sensitivity_dtype,device=w.device)

        if self.use_colreorder :

            col_sums = sensitivity.sum(dim=0)            
            col_order = torch.argsort(col_sums, descending=True) # maybe use descending=False 
            sensitivity = sensitivity[:, col_order] 
            invcol = torch.argsort(col_order)

        if self.use_rowreorder :
            row_sums = sensitivity.sum(dim=1)            
            row_order = torch.argsort(row_sums, descending=True)  # maybe use descending=False 
            sensitivity = sensitivity[row_order, :]   
            invrow = torch.argsort(row_order)


        score = direct_block(sensitivity, self.row_interval, group_size, bit = n_bit)
        score = score.to(w.device)

        if self.use_rowreorder :
            score = score[invrow, :]


        input_feat = input_feat.view(-1, input_feat.shape[-1])

        if self.use_colreorder :
            input_feat  = input_feat[:, col_order.to(input_feat.device)] 


        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
        input_feat = input_feat.to(w.device)

        if self.use_colreorder :
            w = w[:, col_order]

        w = w.reshape(w.shape[0], 1, -1, group_size)

        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = clip_asym_pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config, in_channel = org_w_shape[1] ,score = score[i_b*oc_batch_size:(i_b+1)*oc_batch_size, :])
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        del input_feat
        del org_out
        gc.collect()
        torch.cuda.empty_cache()
        return best_max_val.squeeze(1)

    @torch.no_grad()
    def apply_clip_sym(self, module, clip_list,layer_idx=None):
        for name, max_val in clip_list:
            layer = self.get_op_by_name(module, name)
            layer.cuda()
            max_val = max_val.to(layer.weight.device)
            org_shape = layer.weight.shape

            if layer_idx != None :
                sensitivity = np.load(f'{self.sensitivity_path}/model.layers.{layer_idx}.{name}.npy')
            else :
                sensitivity = np.load(f'{self.sensitivity_path}/{name}.npy')

            sensitivity_dtype = torch.float32
            sensitivity = torch.tensor(sensitivity,dtype=sensitivity_dtype,device=layer.weight.device)
            if self.use_colreorder :
                col_sums = sensitivity.sum(dim=0)            
                col_order = torch.argsort(col_sums, descending=True)  
                sensitivity = sensitivity[:, col_order] 
                invcol = torch.argsort(col_order)
                layer.weight.data = layer.weight.data[:,col_order]


            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)

            if self.use_colreorder :
                layer.weight.data = layer.weight.data[:, invcol]
            layer.cpu()


    @torch.no_grad()
    def apply_awq(self, awq_results, real_quant):
        
        self.apply_scale(self.model, awq_results["scale"])     

        if self.clip_asym:
            self.apply_clip_asym(self.model, awq_results["clip"])
        else:
            self.apply_clip_sym(self.model, awq_results["clip"])

        self.model = self.model.to('cpu')

        layers = self.model.model.layers
        for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
            layer = layers[i]
            named_linears = {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}
            for n, m in named_linears.items():
                m.to(self.dev)
                key = f'model.layers.{i}.{n}'
                if type(self.wbits) is dict :
                    n_bit = self.wbits[key]
                else :
                    n_bit = self.wbits

                if real_quant:

                    intweight,scales,zeros,in_reorder,out_reorder,block_bitwidth = real_quantize_tensor(
                        m.weight.data, n_bit=n_bit, 
                        q_group_size = self.group_size, use_rowreorder=self.use_rowreorder, use_colreorder=self.use_colreorder,
                        sensitivity_path=self.sensitivity_path, linear_name=n, layer_idx = i,
                        row_interval=self.row_interval)
                    

                    new_linear = QuantLinear(
                        bits=n_bit,
                        group_size=self.group_size,
                        outfeature_interval=self.row_interval,
                        infeatures=m.in_features,
                        outfeatures=m.out_features,
                        dtype = self.model.dtype,
                        bias=not m.bias is None
                    )

                    new_linear.pack(in_reorder,out_reorder,intweight,scales,zeros,block_bitwidth,m.bias)
                    new_linear.to(next(layer.parameters()).device)
                    self.set_op_by_name(layer, n, new_linear)
                    new_linear.cpu()
                    del m

                else :
                    m.weight.data = pseudo_quantize_tensor(
                        m.weight.data, n_bit=n_bit,
                        q_group_size = self.group_size, use_rowreorder=self.use_rowreorder, use_colreorder=self.use_colreorder,
                        sensitivity_path=self.sensitivity_path, linear_name=n, layer_idx = i,
                        row_interval=self.row_interval
                    )
                    m.cpu()
    def run(self, nsamples=128, seqlen=512, real_quant = False, awq_results_cache = None, save_path = './awq_quantized_model',):

        if awq_results_cache is None :
            awq_results = self.run_awq(n_samples=nsamples, seqlen=seqlen)
            awq_cache_path = f'awq_cache'
            os.makedirs(awq_cache_path, exist_ok=True)

            if type(self.wbits) == dict :
                torch.save(awq_results, f'{awq_cache_path}/g{self.group_size}.pt')
            else :
                torch.save(awq_results, f'{awq_cache_path}/w{self.wbits}g{self.group_size}.pt')
        else :
            awq_results = torch.load(awq_results_cache,map_location='cpu')
        
        self.load_model(device_map='cpu', dtype=self.dtype)
        self.apply_awq(awq_results,real_quant)
        torch.cuda.empty_cache()
        gc.collect()

        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)


if __name__ == '__main__' :

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )

    parser.add_argument(
        '--sensitivity_path', type=str, default=None,
        help='Path to load sensitivity files from.'
    )

    parser.add_argument(
        '--use_colreorder', action='store_true',
        help='whether use colreorder'
    )

    parser.add_argument(
        '--use_rowreorder', action='store_true',
        help='whether use rowreorder'
    )

    parser.add_argument(
        '--use_bitallocation', action='store_true',
        help='whether use bitallocation'
    )

    parser.add_argument(
        '--clip_asym', action='store_true',
        help='whether to use clip_asym'
    )

    parser.add_argument(
        '--wbits', type=float, default= None,
        help='args.wbits'
    )

    parser.add_argument(
        '--row_interval', type=int, default=32,
        help='row interval for block quantization'
    )

    parser.add_argument(
        '--groupsize', type=int, default= 128,
        help='args.groupsize',choices=[64,128,256,512]
    )

    parser.add_argument(
        '--awq_results_cache', type=str, default= None,
        help='awq results cache path'
    )

    parser.add_argument(
        '--real_quant',action='store_true',
        help='whether use real quant, for example, ./awq_cache/w2.75g128.pt'
    )

    parser.add_argument(
        '--save', default='./awq_quantized_model',
        help='save path for quantized model'
    )

    parser.add_argument(
        '--dtype', default='float16',choices=['float16','bfloat16'],
        help='default dtype for quantized model'
    )

    parser.add_argument(
        '--level', default=3, type=int,
        help='level for bit allocation'
    )

    parser.add_argument(
        '--bit_choices',
        type=int,
        nargs='+',  
        default=[2,3,4],
        help='bit choices for multi-level bit allocation, e.g. --bit_choices 2 3 4'
    )

    args = parser.parse_args()

    if args.use_bitallocation:

        print("====================Using bit allocation======================")
        wbits,bits,quantile = bit_allocation(model_path = args.model_name,sensitivity_path = args.sensitivity_path,bit = args.wbits,
                               use_colreorder=args.use_colreorder, use_rowreorder = args.use_rowreorder,row_interval=args.row_interval,
                               groupsize=args.groupsize,level=args.level, bit_choices=args.bit_choices)
        mixed_precision_config['bits'] = bits
        mixed_precision_config['quantile'] = quantile
        check_mixed_precison_config(mixed_precision_config)
        real_avebits = calculate_average_bit(model_path = args.model_name,sensitivity_path = args.sensitivity_path,bit_allocation = wbits)
        
        print("Average bit:",real_avebits)
        
    else :
        wbits =args.wbits

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else :
        dtype = torch.float32

    awq_model = AWQ(args.model_name, None, None, 'auto', dtype = dtype, group_size=args.groupsize, clip_asym = args.clip_asym, 
                    wbits = wbits, sensitivity_path=args.sensitivity_path, 
                    use_colreorder = args.use_colreorder, use_rowreorder=args.use_rowreorder,
                    row_interval = args.row_interval)

    save_dir = args.save 
    quantization_BPW = 32 / args.groupsize 
    BPW = round(args.wbits + quantization_BPW,2)
    save_dir = f'{save_dir}-wbits{args.wbits}-g{args.groupsize}-BPW{BPW}'

    tick = time.time()
    awq_model.run(nsamples=128,seqlen=512,real_quant = args.real_quant,awq_results_cache=args.awq_results_cache,save_path=save_dir)

    print(f'quantization time : {time.time() - tick} s')



