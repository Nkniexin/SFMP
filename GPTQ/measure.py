import torch



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

    importance = w.reshape(h_blocks, block_h, w_blocks, block_w).sum(dim=(1, 3))

    bit_alloc_block,mask = assign_importance_levels(importance,bit)


    bit_alloc_full = bit_alloc_block.repeat_interleave(block_h, dim=0).to(torch.int)



    return bit_alloc_full , mask 