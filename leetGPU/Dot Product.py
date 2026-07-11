import torch
import triton
import triton.language as tl


@triton.jit
def dot_product_kernel(a_ptr, b_ptr, N, result, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    a = tl.load(a_ptr + offset, mask=mask)
    b = tl.load(b_ptr + offset, mask=mask)
    c = tl.where(mask, a * b, 0)
    local = tl.sum(c)
    tl.atomic_add(result, local)

# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    dot_product_kernel[grid](a, b, N, result, BLOCK_SIZE)
