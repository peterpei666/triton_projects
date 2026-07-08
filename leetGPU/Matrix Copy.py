import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    a_ptr = a_ptr.to(tl.pointer_type(tl.int32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.int32))
    offset0 = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset1 = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask0 = offset0 < N
    mask1 = offset1 < N
    offset = offset0[:, None] * N + offset1[None, :]
    mask = mask0[:, None] & mask1[None, :]
    x = tl.load(a_ptr + offset, mask=mask)
    tl.store(b_ptr + offset, x, mask=mask)
    
# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 64
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    grid = (grid_size, grid_size)
    matmul_kernel[grid](a, b, N, BLOCK_SIZE)
