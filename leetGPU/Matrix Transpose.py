import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    offset0 = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset1 = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_mask = (offset0[:, None] < rows) & (offset1[None, :] < cols)
    in_offset = offset0[:, None] * stride_ir + offset1[None, :] * stride_ic
    input = tl.load(input_ptr + in_offset, mask=in_mask)
    output = input.T
    out_mask = in_mask.T
    out_offset = offset1[:, None] * stride_or + offset0[None, :] * stride_oc
    tl.store(output_ptr + out_offset, output, mask=out_mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1
    BLOCK_SIZE = 16
    grid = (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE))
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        BLOCK_SIZE
    ) 
