"""
dense_kernel.py -- Milestone 1, Task 1.2
i-j-k (naive) vs i-k-j (interchanged) dense matrix multiply on an
M = N/2 sub-block, where N is this student's salted dimension.

Usage: python3 src/dense_kernel.py --student-id <YOUR_ID>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from salt import get_student_params

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def make_matrices(M, seed):
    rng = np.random.default_rng(seed)
    A = rng.random((M, M), dtype=np.float64)
    B = rng.random((M, M), dtype=np.float64)
    return A, B


def _matmul_ijk_py(A, B, M):
    """Naive i-j-k: B[k, j] strides by M elements on the inner (k) step."""
    C = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        Ai = A[i]
        for j in range(M):
            s = 0.0
            for k in range(M):
                s += Ai[k] * B[k, j]
            C[i, j] = s
    return C


def _matmul_ikj_py(A, B, M):
    """Interchanged i-k-j: hoist A[i,k], update row C[i,:] with a
    stride-1 contiguous AXPY against row B[k,:]."""
    C = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        Ci = C[i]
        Ai = A[i]
        for k in range(M):
            a_ik = Ai[k]
            if a_ik != 0.0:
                Ci += a_ik * B[k]
    return C


if HAVE_NUMBA:
    @njit(cache=True)
    def _matmul_ijk_nb(A, B, M):
        C = np.zeros((M, M), dtype=np.float64)
        for i in range(M):
            for j in range(M):
                s = 0.0
                for k in range(M):
                    s += A[i, k] * B[k, j]
                C[i, j] = s
        return C

    @njit(cache=True)
    def _matmul_ikj_nb(A, B, M):
        C = np.zeros((M, M), dtype=np.float64)
        for i in range(M):
            for k in range(M):
                a_ik = A[i, k]
                for j in range(M):
                    C[i, j] += a_ik * B[k, j]
        return C

    matmul_ijk = _matmul_ijk_nb
    matmul_ikj = _matmul_ikj_nb
else:
    matmul_ijk = _matmul_ijk_py
    matmul_ikj = _matmul_ikj_py


def median_time(fn, *args, runs=3):
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--block-size", type=int, default=None,
                     help="Override M directly for quick local testing.")
    args = ap.parse_args()

    params = get_student_params(args.student_id)
    N = params["N"]
    M = args.block_size or (N // 2)

    print(f"N={N} (salted), running sub-block M={M}")
    if not HAVE_NUMBA and M > 300:
        print(f"WARNING: no numba installed; pure-Python O(M^3) with M={M} "
              f"can take tens of minutes. Run 'pip install numba' for a JIT "
              f"version, or pass --block-size to shrink M for a quicker check.")
    A, B = make_matrices(M, params["seed"])
    if HAVE_NUMBA:
        matmul_ijk(A[:8, :8].copy(), B[:8, :8].copy(), 8)  # warm up JIT, excluded from timing
        matmul_ikj(A[:8, :8].copy(), B[:8, :8].copy(), 8)

    t_naive, C_naive = median_time(matmul_ijk, A, B, M, runs=3)
    t_inter, C_inter = median_time(matmul_ikj, A, B, M, runs=3)

    max_abs_diff = float(np.max(np.abs(C_naive - C_inter)))
    speedup = t_naive / t_inter

    result = {
        "student_id": args.student_id,
        "N_salted": N,
        "M_block": M,
        "median_time_naive_ijk_s": t_naive,
        "median_time_interchanged_ikj_s": t_inter,
        "speedup": speedup,
        "max_abs_diff": max_abs_diff,
        "numerically_consistent": max_abs_diff < 1e-9,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/loop_interchange.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\nWrote results/loop_interchange.json")


if __name__ == "__main__":
    main()
