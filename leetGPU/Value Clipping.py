import torch
import triton
import triton.language as tl


@triton.jit
def clip_kernel(input_ptr, output_ptr, lo, hi, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask)
    y = tl.minimum(tl.maximum(x, lo), hi)
    tl.store(output_ptr + offset, y, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, lo: float, hi: float, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    clip_kernel[grid](input, output, lo, hi, N, BLOCK_SIZE=BLOCK_SIZE)
