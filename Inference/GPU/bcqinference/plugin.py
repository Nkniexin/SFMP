import torch
import custom_kernel

@torch.library.custom_op("plugin::anybcq_gemv", mutates_args={"output"})
def anybcq_gemv(
    x: torch.Tensor, 
    output: torch.Tensor, 
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    q_bias: torch.Tensor, 
    block_bitwidth: torch.Tensor,
    offset: torch.Tensor,
    group_size: int, outfeature_interval: int) -> None:
    custom_kernel.anybcq_gemv(
        x, output, q_weight, alpha, q_bias, block_bitwidth, offset, group_size, outfeature_interval)

# @torch.library.custom_op("plugin::anybcq_dequant", mutates_args=())
def anybcq_dequant(
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    q_bias: torch.Tensor, 
    block_bitwidth: torch.Tensor,
    offset: torch.Tensor,
    group_size: int, outfeature_interval: int) -> torch.Tensor:
    weight = custom_kernel.anybcq_dequant(
        q_weight, alpha, q_bias,block_bitwidth,offset, group_size, outfeature_interval)
    return weight

@anybcq_gemv.register_fake
def _(x, output, q_weight, alpha, q_bias,block_bitwidth,offset, group_size, outfeature_interval):
    return None
