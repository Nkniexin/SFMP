import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig
import numpy as np
import os

from tqdm import tqdm

def check_mixed_precison_config(config: dict={}) :
    assert config["bits"] is not None, "bits must be set"
    assert config["quantile"] is not None, "quantile must be set"

    bits = config["bits"]
    quantile = config["quantile"]

    num_bits = len(bits)
    num_quantiles = len(quantile)
    assert num_bits == num_quantiles + 1, "num_bits must be equal to num_quantiles + 1"

    #check bits and quantile are sorted
    assert bits == sorted(bits), "bits must be sorted"
    assert quantile == sorted(quantile), "quantile must be sorted"

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

def get_average_bit(
    w,
    block_h,
    block_w,
    bits,
    quantile,
    use_rowreorder=True,
    use_colreorder=True
):
    h_blocks = w.shape[0] // block_h
    w_blocks = w.shape[1] // block_w

    if use_colreorder:
        col_sums = w.sum(dim=0)
        col_order = torch.argsort(col_sums, descending=True)
        w = w[:, col_order]

    if use_rowreorder:
        row_sums = w.sum(dim=1)
        row_order = torch.argsort(row_sums, descending=True)
        w = w[row_order, :]

    importance = w.reshape(
        h_blocks, block_h, w_blocks, block_w
    ).sum(dim=(1, 3)).to(torch.float32)

    levels = torch.full_like(importance, fill_value=bits[0])

    num_bits = len(bits)
    num_quantile = len(quantile)

    assert num_bits == num_quantile + 1, "num_bits must be equal to num_quantile + 1"

    for i in range(num_quantile) :
        i_quantile = quantile[i]
        levels[importance > i_quantile] = bits[i+1]

    return levels.mean()


def bit_allocation_2level(model_path, sensitivity_path, bit, row_interval = None,groupsize = None, use_rowreorder:bool = True, use_colreorder:bool = True) :
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

    bits = [base_bit, base_bit + 1]
    quantile = [q]

    bit_allocation = {}
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:            
            sensitivity = np.load(f'{sensitivity_path}/{name}.npy')
            sensitivity_dtype = torch.float32
            sensitivity = torch.from_numpy(sensitivity).to(sensitivity_dtype)

            average_bit = get_average_bit(sensitivity,row_interval,groupsize,bits,quantile,use_rowreorder=use_rowreorder, use_colreorder=use_colreorder)

            bit_allocation[name] = average_bit.item()

    with open(f'bit_allocation.json','w') as f:
        
        import json

        json.dump(bit_allocation,f,indent=4)

    del model

    return bit_allocation, bits, quantile


def bit_allocation_3level(model_path, sensitivity_path, bit, row_interval = None,groupsize = None, use_rowreorder:bool = True, use_colreorder:bool = True,bit_choices: list = None) :
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

    if bit_choices is None:
        bits = [2,3,4] # default bits choices
    else:
        bits = bit_choices

    b1, b2, b3 = bits
    assert b1 < b2 < b3, "bits must be sorted"

    # binary search to find the best bit allocation
    importance_sorted, _ = torch.sort(importance)

    N = importance_sorted.numel()

    # loss scaling
    def loss_scale(b):
        return (2**b - 1) ** 3

    loss_2 = loss_scale(bits[0])
    loss_3 = loss_scale(bits[1])
    loss_4 = loss_scale(bits[2])

    results = []
    step = 0.01

    alpha1_list = np.arange(0.0, 1.0, step)

    for alpha1 in alpha1_list:

        alpha3 = (bit - b2 - (b1 - b2) * alpha1) / (b3 - b2)
        alpha2 = 1 - alpha1 - alpha3

        if not (0 <= alpha2 <= 1 and 0 <= alpha3 <= 1):
            continue

        n1 = int(alpha1 * N)
        n2 = int(alpha2 * N)
        n3 = N - n1 - n2

        if n3 < 0:
            continue

        part1 = importance_sorted[:n1]               # 2-bit
        part2 = importance_sorted[n1:n1+n2]          # 3-bit
        part3 = importance_sorted[n1+n2:]            # 4-bit

        total_loss = 0.0

        if n1 > 0:
            total_loss += (part1 / loss_2).sum()
        if n2 > 0:
            total_loss += (part2 / loss_3).sum()
        if n3 > 0:
            total_loss += (part3 / loss_4).sum()

        results.append((alpha1, alpha2, alpha3, total_loss.item()))

    best = min(results, key=lambda x: x[3])
    alpha1, alpha2, alpha3, _ = best

    q1 = torch.quantile(importance, alpha1)
    q2 = torch.quantile(importance, alpha1 + alpha2)

    quantile = [q1,q2]

    bit_allocation = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:

            sensitivity = np.load(f'{sensitivity_path}/{name}.npy')
            sensitivity = torch.from_numpy(sensitivity).to(torch.float32)

            avg_bit = get_average_bit(
                sensitivity,
                row_interval,
                groupsize,
                bits,
                quantile,
                use_rowreorder=use_rowreorder,
                use_colreorder=use_colreorder
            )

            bit_allocation[name] = avg_bit.item()
    
    with open(f'bit_allocation.json','w') as f:
        
        import json

        json.dump(bit_allocation,f,indent=4)

    del model

    return bit_allocation, bits, quantile


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

def bit_allocation(model_path, sensitivity_path, bit, row_interval = None,groupsize = None, use_rowreorder:bool = True, use_colreorder:bool = True,level:int = 3,bit_choices: list = None) :

    if level == 2:
        return bit_allocation_2level(model_path, sensitivity_path, bit, row_interval,groupsize, use_rowreorder, use_colreorder)
    elif level == 3:
        assert bit_choices is not None and len(bit_choices) == 3, "bit_choices must be a list of 3 bits"
        return bit_allocation_3level(model_path, sensitivity_path, bit, row_interval,groupsize, use_rowreorder, use_colreorder, bit_choices)
    else:
        raise ValueError(f"level {level} is not supported")




    







            
            