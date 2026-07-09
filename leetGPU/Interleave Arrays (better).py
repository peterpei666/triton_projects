import torch
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    a = tl.load(A_ptr + offset, mask=mask)
    b = tl.load(B_ptr + offset, mask=mask)
    res = tl.interleave(a, b)
    out_offset = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
    out_mask = out_offset < N * 2
    tl.store(output_ptr + out_offset, res, mask=out_mask)

# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    interleave_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)
