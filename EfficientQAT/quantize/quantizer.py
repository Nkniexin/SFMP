import torch
import torch.nn as nn
import gc
import numpy as np

import pdb
from itertools import product

import torch.optim as optim

CLIPMIN = 1e-4

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


def assign_importance_levels(importance,bit :float = 3.5):

    flat = importance.flatten()
    
    base_bit = int(bit)
    percent = bit - base_bit
    q = torch.quantile(flat, 1.0 - percent)

    levels = torch.full_like(importance, fill_value=base_bit)

    levels[importance > q] = base_bit + 1

    mask = torch.where(levels == (base_bit + 1), 1, 0)
    
    return levels , mask
def direct_block(w,block_h:int = 32,block_w :int = 128,bit : float = 3.5):

    h_blocks = w.shape[0] // block_h
    w_blocks = w.shape[1] // block_w

    importance = w.reshape(h_blocks, block_h, w_blocks, block_w).sum(dim=(1, 3)).to(torch.float32)

    bit_alloc_block,mask = assign_importance_levels(importance,bit)

    bit_alloc_full = bit_alloc_block.repeat_interleave(block_h, dim=0).to(torch.int)

    return bit_alloc_full , mask 


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits:float = 3.0,
        group_size=None,
        weight=None,
        sensitivity_path=None,
        layer_idx=None,
        linear_name = None,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        
        score, col_order, invcol = self.bit_allocation(sensitivity_path, layer_idx, linear_name, weight.dtype, n_bits, group_size,weight.device)
        self.register_buffer('score',score)
        self.register_buffer('col_order',col_order)
        self.register_buffer('invcol',invcol)
        assert weight.dim() == 2
        weight = weight[:,self.col_order]

        self.score = self.score.reshape(-1,1)
        self.register_buffer('qmax', 2 ** (self.score) - 1)
        self.register_buffer('qmin', torch.zeros_like(self.qmax))

        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1,self.group_size)
                xmin = x.amin([-1], keepdim=True)
                xmax =  x.amax([-1], keepdim=True)
                range = xmax - xmin
                scale = range / self.qmax
                scale = scale.clamp(min=1e-4, max=1e4)
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())
    def bit_allocation(self,sensitivity_path = None, layer_idx = None, linear_name = None,dtype = None, wbits = None, groupsize = None,device = None) :

        row_interval = 32
        sensitivity = np.load(f'{sensitivity_path}/model.layers.{layer_idx}.{linear_name}.npy')
        sensitivity = torch.tensor(sensitivity, dtype=dtype, device = device)

        col_sums = sensitivity.sum(dim=0)            
        col_order = torch.argsort(col_sums, descending=True)  # maybe use descending=False 
        sensitivity = sensitivity[:, col_order] 
        invcol = torch.argsort(col_order)

        row_sums = sensitivity.sum(dim=1)            
        row_order = torch.argsort(row_sums, descending=True)  # maybe use descending=False 
        sensitivity = sensitivity[row_order, :]   
        invrow = torch.argsort(row_order)

        score, mask = direct_block(sensitivity, row_interval, groupsize, bit = wbits)
        score  = score[invrow, :]

        return score,col_order,invcol
            

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)
        
    def fake_quant(self, x):

        x = x[:, self.col_order]
        scale = clamp_ste(self.scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        
        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)

        x_dequant = x_dequant[:, self.invcol]
        return x_dequant
    
    def get_intvalue(self,x: torch.Tensor) :

        x = x[:, self.col_order]
        scale = clamp_ste(self.scale,1e-4, 1e4)
        
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        
        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)

        x_int = x_int.to(torch.uint8)
        x_int = x_int.reshape(dim1, dim2)

        x_int = x_int[:, self.invcol]
        return x_int

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        
        x_dequant = self.fake_quant(x)
        return x_dequant

