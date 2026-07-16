import torch
import triton
import triton.language as tl


@triton.jit
def prefix_sum_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    total = 0.0
    for base in range(0, N, BLOCK_SIZE):
        offset = base + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        x = tl.load(input_ptr + offset, mask=mask)
        temp = tl.cumsum(x)
        tl.store(output_ptr + offset, temp + total, mask=mask)
        total += tl.sum(x)

# data and output are tensors on the GPU
def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    BLOCK_SIZE = 1024
    grid = (1,)
    prefix_sum_kernel[grid](data, output, n, BLOCK_SIZE)
