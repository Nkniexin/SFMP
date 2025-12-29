import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig
import numpy as np
import os

from tqdm import tqdm

def get_block_importance(w,block_h:int = 32,block_w :int = 128, use_rowreorder:bool = True, use_colreorder:bool = True):

    h_blocks = w.shape[0] // block_h
    w_blocks = w.shape[1] // block_w

    if use_colreorder :
        col_sums = w.sum(dim=0)            
        col_order = torch.argsort(col_sums, descending=True)   
        w = w[:, col_order] 
    if use_rowreorder :
        row_sums = w.sum(dim=1)            
        row_order = torch.argsort(row_sums, descending=True)  
        w = w[row_order, :]   

    importance = w.reshape(h_blocks, block_h, w_blocks, block_w).sum(dim=(1, 3)).to(torch.float32)

    return importance

def get_average_bit(w,block_h,block_w,q,bit,use_rowreorder:bool = True, use_colreorder:bool = True) :

    h_blocks = w.shape[0] // block_h
    w_blocks = w.shape[1] // block_w

    base_bit = int(bit)

    if use_colreorder :
        col_sums = w.sum(dim=0)            
        col_order = torch.argsort(col_sums, descending=True)  
        w = w[:, col_order] 
    if use_rowreorder :
        row_sums = w.sum(dim=1)            
        row_order = torch.argsort(row_sums, descending=True)  
        w = w[row_order, :]   

    importance = w.reshape(h_blocks, block_h, w_blocks, block_w).sum(dim=(1, 3)).to(torch.float32)

    levels = torch.full_like(importance, fill_value=base_bit)

    levels[importance > q] = base_bit + 1

    average_bit = levels.mean()

    return average_bit

def bit_allocation(model_path, sensitivity_path, bit, row_interval = None,groupsize = None, use_rowreorder:bool = True, use_colreorder:bool = True) :
    importance = []
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            dtype=config.dtype,
            trust_remote_code=True,        
        )
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            sensitivity = np.load(f'{sensitivity_path}/{name}.npy')
            sensitivity_dtype = torch.float32
            sensitivity = torch.from_numpy(sensitivity).to(sensitivity_dtype)

            block_importance = get_block_importance(sensitivity,row_interval,groupsize,use_rowreorder=use_rowreorder, use_colreorder=use_colreorder)

            block_importance = block_importance.flatten()

            importance.append(block_importance)

    importance = torch.cat(importance)

    importance = importance.flatten()

    base_bit = int(bit)
    percent = bit - base_bit

    q = torch.quantile(importance, 1.0 - percent)

    bit_allocation = {}
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:            
            sensitivity = np.load(f'{sensitivity_path}/{name}.npy')
            sensitivity_dtype = torch.float32
            sensitivity = torch.from_numpy(sensitivity).to(sensitivity_dtype)

            average_bit = get_average_bit(sensitivity,row_interval,groupsize,q,bit,use_rowreorder=use_rowreorder, use_colreorder=use_colreorder)

            bit_allocation[name] = average_bit.item()

    with open(f'bit_allocation.json','w') as f:
        
        import json

        json.dump(bit_allocation,f,indent=4)

    del model

    return bit_allocation


def calculate_average_bit(model_path, sensitivity_path,bit_allocation) :

    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            trust_remote_code=True,
            dtype=config.dtype,        
        )

    bit_sum = 0.0
    param_sum = 0
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            sensitivity = np.load(f'{sensitivity_path}/{name}.npy')
            bit_sum += bit_allocation[name]*sensitivity.size
            param_sum += sensitivity.size

    average_bit = bit_sum / param_sum

    del model
    return average_bit



    







            
            