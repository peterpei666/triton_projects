import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(input_ptr, kernel_ptr, output_ptr, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in tl.range(kernel_size):
        input = tl.load(input_ptr + offset + i, mask=(offset + i < input_size))
        kernel = tl.load(kernel_ptr + i)
        output += input * kernel
    tl.store(output_ptr + offset, output, mask=(offset < input_size - kernel_size + 1))

# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE)
