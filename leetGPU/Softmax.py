import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel(input_ptr, max_element, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask, other=float("-inf"))
    local_max = tl.max(x, axis=0)
    tl.atomic_max(max_element, local_max)

@triton.jit
def sum_exp_kernel(input_ptr, max_element, total, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask, other=0.0)
    m = tl.load(max_element)
    t = tl.where(mask, tl.exp(x - m), 0.0)
    local_sum = tl.sum(t, axis=0)
    tl.atomic_add(total, local_sum)

@triton.jit
def softmax_kernel(input_ptr, output_ptr, max_element, total, N, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask, other=0.0)
    m = tl.load(max_element)
    s = tl.load(total)
    val = tl.exp(x - m) / s
    output = tl.where(mask, val, 0.0)
    tl.store(output_ptr + offset, output, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    max_element = torch.zeros(1, 1, device=input.device)
    total = torch.zeros(1, 1, device=input.device)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    max_kernel[grid](input, max_element, N, BLOCK_SIZE)
    sum_exp_kernel[grid](input, max_element, total, N, BLOCK_SIZE)
    softmax_kernel[grid](input, output, max_element, total, N, BLOCK_SIZE)
