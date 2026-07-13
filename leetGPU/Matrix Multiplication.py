import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ans = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, N, BLOCK_K):
        offset_k = k + tl.arange(0, BLOCK_K)
        a_off = offset_m[:, None] * stride_am + offset_k[None, :] * stride_an
        b_off = offset_k[:, None] * stride_bn + offset_n[None, :] * stride_bk
        a_mask = (offset_m[:, None] < M) & (offset_k[None, :] < N)
        b_mask = (offset_k[:, None] < N) & (offset_n[None, :] < K)
        a = tl.load(a_ptr + a_off, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + b_off, mask=b_mask, other=0.0)
        ans = tl.dot(a, b, ans)
    c_off = offset_m[:, None] * stride_cm + offset_n[None, :] * stride_ck
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < K)
    tl.store(c_ptr + c_off, ans, mask=c_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    grid = (triton.cdiv(M,BLOCK_M), triton.cdiv(K,BLOCK_K))
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
