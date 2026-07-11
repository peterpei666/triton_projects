import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(input_ptr, histogram, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask)
    tl.atomic_add(histogram + x, 1, mask=mask)

# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    histogram_kernel[grid](input, histogram, N, BLOCK_SIZE)
