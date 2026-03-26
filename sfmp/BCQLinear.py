"""
This script converts a given model (QuantLinear) into the BCQ (BCQLinear) format.
It loads the model, applies the required quantization procedures, and exports the final BCQ
weights and metadata.
"""

import torch
import json
import torch.nn as nn
from tqdm import tqdm
from .QuantLinear import QuantLinear,load_quantized_model
import math
from .utils import *

DeBug = False
def pack_bits_to_int32(x: torch.Tensor):
    """
    x: tensor of shape [dim0, dim1, dim2], values ∈ {0,1}
    Returns a tensor of shape [dim0//32, dim1, dim2], dtype=int32.
    Each output element is a packed 32-bit integer.
    """
    assert x.shape[0] % 32 == 0, "dim0 must be divisible by 32"

    x_int = x.to(torch.int32)
    b, h, w = x.shape
    x_int = x_int.view(b // 32, 32, h, w)   

    shifts = torch.arange(32, device=x.device, dtype=torch.int32).view(32, 1, 1)

    masks = 1 << shifts   

    packed = torch.sum(x_int * masks, dim=1)

    return packed.to(torch.int32)

class BCQLinear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        wbits, 
        group_size,
        qweight_metadata_len = None,
        outfeature_interval = None,
        bias=False, 
        dtype=torch.half):
        super().__init__()


        assert wbits <= 4.0 and  type(wbits) == float,'wbits must be less than 4.0 and float type'
        self.in_features = in_features
        self.out_features = out_features
        self.wbits = wbits
        if group_size == -1:
            self.group_size = in_features
        else:
            self.group_size = group_size
        self.dtype = dtype

        self.outfeature_interval = outfeature_interval

        self.register_buffer(
            'in_reorder',
            torch.zeros((in_features,), dtype=torch.int32)
        )

        self.register_buffer(
            'out_reorder',
            torch.zeros((out_features,), dtype=torch.int32)
        )

        self.register_buffer(
            'qweight',
            torch.empty(
                qweight_metadata_len, 
                dtype=torch.int32)
        )

        self.register_buffer(
            "alpha",
            torch.empty(
                (in_features // self.group_size, out_features), 
                dtype=dtype)
        )

        self.register_buffer(
            "beta",
            torch.empty(
                (in_features // self.group_size, out_features), 
                dtype=dtype)
        )

        self.register_buffer(
            'block_bitwidth',
            torch.zeros((in_features // self.group_size, out_features // outfeature_interval), dtype=torch.int8)
        )

        self.register_buffer(
            'offset',
            torch.zeros(in_features // self.group_size * out_features // outfeature_interval, dtype=torch.int32)
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=self.dtype)
            )
        else:
            self.bias = None
    def pack_from_quantLinear(self, quantLinear : QuantLinear) :
        
        intweight = unpack_int4(quantLinear.intweight)
        alpha = quantLinear.scales.to(self.dtype)
        beta = quantLinear.zeros.to(self.dtype)

        if DeBug : 
            torch.save(intweight,'quantlinear_intweight.pt')
            torch.save(alpha,'quantlinear_alpha.pt')
            torch.save(beta,'quantlinear_beta.pt')

        

        group_size = quantLinear.group_size
        outfeature_interval = quantLinear.outfeature_interval

        assert self.in_features % group_size == 0
        assert self.out_features % outfeature_interval == 0

        block_bitwidth = quantLinear.block_bitwidth
        qweight_metadata_len = block_bitwidth.sum().item() * outfeature_interval * group_size / 32 # using torch.int32 to store

        if quantLinear.bias is not None:
            bias = quantLinear.bias.to(self.dtype)
        else :
            bias = None

        intweight = ((intweight.unsqueeze(-1) >> torch.arange(4, device=intweight.device)) & 1).to(torch.uint8)

        qweight = []

        for i in range(0, self.in_features, group_size) :
            for j in range(0, self.out_features, outfeature_interval) :

                n_bit = block_bitwidth[i // group_size][j // outfeature_interval].item()
                intweight_ = intweight[i:i+group_size,j:j+outfeature_interval]
                intweight_ = intweight_[:,:,:n_bit]

                intweight_ = intweight_.permute(0,2,1)
                intweight_ = pack_bits_to_int32(intweight_)
                intweight_ = intweight_.flatten()
                qweight.append(intweight_)

        qweight = torch.cat(qweight)

        if DeBug :
            torch.save(qweight,'bcq_qweight.pt')

        

        # print('quantized weight shape', qweight.shape)

        assert qweight.numel() == qweight_metadata_len


        block_extra = block_bitwidth.repeat_interleave(outfeature_interval, dim=1).to(torch.int)

        beta = -beta * alpha   # TODO: maybe need to use float to reduce error.
        # print('block_extra :', block_extra.shape)

        sum_alpha = alpha * ( (1 << block_extra) - 1 )

        beta = beta + sum_alpha / 2

        alpha = alpha /2 

        if DeBug :
            torch.save(alpha,'bcq_alpha.pt')
            torch.save(beta,'bcq_beta.pt')
        self.qweight = qweight.clone()
        self.alpha = alpha.clone().to(self.dtype)
        self.beta = beta.clone().to(self.dtype)

        self.in_reorder = quantLinear.in_reorder.clone()
        self.out_reorder = quantLinear.out_reorder.clone()


        flatten_bitwidth = block_bitwidth.flatten().cpu().numpy()
        offset = []
        for i in range(flatten_bitwidth.shape[0]) :
            if i == 0 :
                offset.append(0)
            else :
                offset.append(offset[i-1] + flatten_bitwidth[i-1]*outfeature_interval*group_size // 32)

        self.block_bitwidth = block_bitwidth.to(torch.int8).clone()
        self.offset = torch.tensor(offset).to(torch.int32)

        if DeBug :

            torch.save(self.block_bitwidth,'bcq_block_bitwidth.pt')
            torch.save(self.offset,'bcq_offset.pt')

            exit(0)
        
        if bias is not None :
            self.bias = bias.clone().to(self.dtype)
        else :
            self.bias = None


@torch.no_grad()
def export_bcq(model) :

    layers = model.model.layers
    for i in tqdm(range(len(layers))) :
        layer = layers[i]
        named_linears = get_named_linears(layer, QuantLinear)
        for name, module in named_linears.items() :
                
                bcq_linear = BCQLinear(module.infeatures, module.outfeatures, module.bits, module.group_size, module.qweight_metadata_len.item() ,module.outfeature_interval, 
                                       module.bias if module.bias is not None else False,dtype=torch.float16)  # TODO: dtype maybe not float16

                bcq_linear.pack_from_quantLinear(module)

                set_op_by_name(layer,name,bcq_linear)

    return model







    




        
        









        

