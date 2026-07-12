import torch
import triton
import triton.language as tl


@triton.jit
def mse_kernel(a_ptr, b_ptr, mse, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    a = tl.load(a_ptr + offset, mask=mask, other=0.0)
    b = tl.load(b_ptr + offset, mask=mask, other=0.0)
    dif = a - b
    local = tl.sum(dif * dif) / N
    tl.atomic_add(mse, local)

# predictions, targets, mse are tensors on the GPU
def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    mse_kernel[grid](predictions, targets, mse, N, BLOCK_SIZE)
