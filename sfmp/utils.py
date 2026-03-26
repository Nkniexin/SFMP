import logging
from termcolor import colored
import torch
import sys
import os
import time
def get_named_linears(module, type):
    # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def pack_int4(t: torch.Tensor) -> torch.Tensor:
    """
    t: int8 tensor of shape [dim0, dim1]
       each element uses only lower 4 bits
    return: int8 tensor of shape [dim0//2, dim1]
    """
    assert t.dtype == torch.int8
    assert t.dim() == 2
    assert t.size(0) % 2 == 0

    t_u = t.to(torch.uint8) & 0x0F

    low  = t_u[0::2]        # [dim0/2, dim1]
    high = t_u[1::2] << 4   # [dim0/2, dim1]

    packed = low | high
    return packed.to(torch.int8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    packed: int8 tensor of shape [dim0//2, dim1]
    return: int8 tensor of shape [dim0, dim1]
    """
    assert packed.dtype == torch.int8
    assert packed.dim() == 2

    p = packed.to(torch.uint8)

    low  = p & 0x0F
    high = (p >> 4) & 0x0F

    dim0 = packed.size(0) * 2
    dim1 = packed.size(1)

    out = torch.empty((dim0, dim1), dtype=torch.int8, device=packed.device)
    out[0::2] = low.to(torch.int8)
    out[1::2] = high.to(torch.int8)

    return out