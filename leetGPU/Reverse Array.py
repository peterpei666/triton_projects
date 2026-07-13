import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(input_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    p1 = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = p1 < N // 2
    p2 = N - p1 - 1
    a = tl.load(input_ptr + p1, mask=mask)
    b = tl.load(input_ptr + p2, mask=mask)
    tl.store(input_ptr + p1, b, mask=mask)
    tl.store(input_ptr + p2, a, mask=mask)


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](input, N, BLOCK_SIZE)
