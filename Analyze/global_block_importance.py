import torch

from transformers import AutoModelForCausalLM

import os

from bit_allocation import bit_alloc

def get_block_importance(w,block_h:int = 32,block_w :int = 128):

    h_blocks = w.shape[0] // block_h
    w_blocks = w.shape[1] // block_w


    col_sums = w.sum(dim=0)            
    col_order = torch.argsort(col_sums, descending=True)  # maybe use descending=False 
    w = w[:, col_order] 

    row_sums = w.sum(dim=1)            
    row_order = torch.argsort(row_sums, descending=True)  # maybe use descending=False 
    w = w[row_order, :]   

    importance = w.reshape(h_blocks, block_h, w_blocks, block_w).sum(dim=(1, 3)).to(torch.float32)

    return importance

def get_average_bit(w,block_h,block_w,q,bit) :

    h_blocks = w.shape[0] // block_h
    w_blocks = w.shape[1] // block_w

    base_bit = int(bit)


    col_sums = w.sum(dim=0)            
    col_order = torch.argsort(col_sums, descending=True)  # maybe use descending=False 
    w = w[:, col_order] 

    row_sums = w.sum(dim=1)            
    row_order = torch.argsort(row_sums, descending=True)  # maybe use descending=False 
    w = w[row_order, :]   

    importance = w.reshape(h_blocks, block_h, w_blocks, block_w).sum(dim=(1, 3)).to(torch.float32)

    levels = torch.full_like(importance, fill_value=base_bit)

    levels[importance > q] = base_bit + 1

    average_bit = levels.mean()

    return average_bit

    

    




if '__main__' == __name__:

    import argparse

    parser = argparse.ArgumentParser()

    bit = 2.25
    impotance_dict = {}

    parser.add_argument(

        '--sensitivity', type=str, default=None,

        help='sensitivity path.'

    )

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(

        args.sensitivity,

        torch_dtype=torch.bfloat16,  

        device_map="auto"           

    )

    model_name = model.config._name_or_path.split('/')[-1]

    importance = []

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            sensitivity_dtype = torch.float32
            sensitivity = module.weight.to(sensitivity_dtype)

            block_importance = get_block_importance(sensitivity,32,128)

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

            
            sensitivity_dtype = torch.float32
            sensitivity = module.weight.to(sensitivity_dtype)

            average_bit = get_average_bit(sensitivity,32,128,q,bit)

            bit_allocation[name] = average_bit.item()

    with open(f'bit_allocation.json','w') as f:
        
        import json

        json.dump(bit_allocation,f,indent=4)







            
            