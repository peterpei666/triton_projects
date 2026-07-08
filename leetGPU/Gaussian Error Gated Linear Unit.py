import torch
import triton
import triton.language as tl
import math

@triton.jit
def geglu(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N // 2
    x1 = tl.load(input_ptr + offset, mask=mask)
    x2 = tl.load(input_ptr + offset + N // 2, mask=mask)
    y = 0.5 * x1 * x2 * (1 + tl.erf(x2 * 0.707106781))
    tl.store(output_ptr + offset, y, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    geglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
