import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(input_ptr, output_ptr, gamma, beta, eps, N, BLOCK_SIZE: tl.constexpr):
    sq_sum = 0.0
    for base in range(0, N, BLOCK_SIZE):
        offset = base + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        x = tl.load(input_ptr + offset, mask=mask, other=0.0)
        sq_sum += tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    rms_inv = tl.rsqrt(sq_sum / N + eps)
    for base in range(0, N, BLOCK_SIZE):
        offset = base + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        x = tl.load(input_ptr + offset, mask=mask, other=0.0)
        y = x * rms_inv * gamma + beta
        tl.store(output_ptr + offset, y, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = 1024
    grid = (1,)
    rms_norm_kernel[grid](input, output, gamma, beta, eps, N, BLOCK_SIZE)
