"""
scaling_study.py -- Milestone 3
Strong scaling (Amdahl) + weak scaling (Gustafson-Barsis) study on a 2D
5-point stencil update, plus a Roofline arithmetic-intensity classification.

Usage: python3 src/scaling_study.py --student-id <YOUR_ID>
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from salt import get_student_params

STENCIL_ITERS = 20  # sweeps per timing, kept modest to fit the time budget


def stencil_chunk(args):
    """One 5-point Jacobi stencil sweep over a row-block, including the
    1-row halo needed from neighboring blocks (serialized here; this
    isolates compute-side scaling rather than a full halo-exchange)."""
    grid, row_start, row_end, iters = args
    local = grid[max(row_start - 1, 0):min(row_end + 1, grid.shape[0])].copy()
    for _ in range(iters):
        new = local.copy()
        new[1:-1, 1:-1] = 0.25 * (
            local[:-2, 1:-1] + local[2:, 1:-1] + local[1:-1, :-2] + local[1:-1, 2:]
        )
        local = new
    return local[1:-1] if row_start > 0 else local[: row_end - row_start]


def run_stencil(N, P, iters):
    grid = np.random.default_rng(0).random((N, N), dtype=np.float64)
    rows_per = max(N // P, 1)
    chunks = []
    for p in range(P):
        rs = p * rows_per
        re_ = N if p == P - 1 else (p + 1) * rows_per
        chunks.append((grid, rs, re_, iters))

    t0 = time.perf_counter()
    if P == 1:
        results = [stencil_chunk(chunks[0])]
    else:
        with mp.Pool(P) as pool:
            results = pool.map(stencil_chunk, chunks)
    elapsed = time.perf_counter() - t0
    return elapsed, sum(r.size for r in results)


def amdahl_s(P, T1, TP):
    return T1 / TP


def extract_s(P, Sp):
    if P == 1:
        return None
    return ((1 / Sp) - (1 / P)) / (1 - (1 / P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--max-procs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    params = get_student_params(args.student_id)
    N = params["N"]
    max_p = max(1, args.max_procs)
    p_values = sorted({1, 2, 4, min(8, max_p), max_p})
    p_values = [p for p in p_values if p <= max_p]

    os.makedirs("results", exist_ok=True)

    # ---- Strong scaling (Amdahl): fixed N across all P ----
    print(f"Strong scaling: N={N} fixed, P in {p_values}")
    strong = {}
    for P in p_values:
        elapsed, _ = run_stencil(N, P, STENCIL_ITERS)
        strong[P] = elapsed
        print(f"  P={P:<2d} T={elapsed:.4f}s")

    T1 = strong[1]
    strong_rows = []
    for P in p_values:
        Sp = amdahl_s(P, T1, strong[P])
        Ep = Sp / P * 100
        s_frac = extract_s(P, Sp)
        strong_rows.append((P, N, strong[P], Sp, Ep, s_frac))

    s_estimates = [r[5] for r in strong_rows if r[5] is not None]
    s_hat = sorted(s_estimates)[len(s_estimates) // 2] if s_estimates else None
    s_max = (1 / s_hat) if s_hat else None

    # ---- Weak scaling (Gustafson): N_P = ceil(N * sqrt(P)) ----
    print(f"\nWeak scaling: N scaled by sqrt(P), P in {p_values}")
    weak = {}
    weak_N = {}
    for P in p_values:
        N_p = int(np.ceil(N * (P ** 0.5)))
        weak_N[P] = N_p
        elapsed, _ = run_stencil(N_p, P, STENCIL_ITERS)
        weak[P] = elapsed
        print(f"  P={P:<2d} N_P={N_p} T={elapsed:.4f}s")

    T1_weak = weak[1]
    weak_rows = []
    for P in p_values:
        work_ratio = (weak_N[P] ** 2) / (N ** 2)  # Work(P)/Work(1)
        scaled_speedup = (P * T1_weak) / (work_ratio * weak[P]) if weak[P] else None
        weak_rows.append((P, weak_N[P], weak[P], scaled_speedup))

    with open("results/scaling_metrics.tsv", "w") as f:
        f.write("mode\tP\tN\ttime_s\tspeedup\tefficiency_or_scaled_speedup\tamdahl_s_estimate\n")
        for P, N_, T_, Sp, Ep, sfrac in strong_rows:
            f.write(f"strong\t{P}\t{N_}\t{T_:.6f}\t{Sp:.4f}\t{Ep:.2f}\t{'' if sfrac is None else f'{sfrac:.6f}'}\n")
        for P, N_, T_, ss in weak_rows:
            f.write(f"weak\t{P}\t{N_}\t{T_:.6f}\t\t{'' if ss is None else f'{ss:.4f}'}\t\n")

    print("\nWrote results/scaling_metrics.tsv")
    print(f"Median Amdahl serial fraction s = {s_hat}")
    print(f"Theoretical max speedup S_max = 1/s = {s_max}")

    # ---- Roofline: arithmetic intensity of the 5-point stencil ----
    # Per interior point per sweep: 4 adds + 1 mul = 5 FLOPs.
    # Naive DRAM traffic (no cache reuse model): 5 reads + 1 write = 6 * 8 bytes.
    flops_per_point = 5
    bytes_per_point_naive = 6 * 8
    ai_naive = flops_per_point / bytes_per_point_naive

    host_profile_path = "results/host_profile.json"
    cpu_info = {}
    if os.path.isfile(host_profile_path):
        cpu_info = json.load(open(host_profile_path)).get("cpu", {})

    roofline = {
        "flops_per_point_per_sweep": flops_per_point,
        "bytes_per_point_naive_traffic": bytes_per_point_naive,
        "arithmetic_intensity_flops_per_byte": ai_naive,
        "note": (
            "AI ~0.10 FLOP/byte is far left on the roofline (typical CPU "
            "ridge point is 4-10+ FLOP/byte), so this naive stencil is "
            "Memory-Bandwidth Bound, not Compute Bound, regardless of core "
            "count -- consistent with sublinear strong-scaling observed above."
        ),
        "regime": "memory_bandwidth_bound",
        "cpu_model": cpu_info.get("model_name"),
        "l3_cache": cpu_info.get("l3_cache"),
    }
    with open("results/roofline_summary.json", "w") as f:
        json.dump(roofline, f, indent=2)
    print("Wrote results/roofline_summary.json")

    # also keep a compact amdahl/gustafson summary for the report/analysis
    with open("results/scaling_summary.json", "w") as f:
        json.dump(
            {
                "N_salted": N,
                "p_values": p_values,
                "amdahl_s_median": s_hat,
                "amdahl_s_max_speedup": s_max,
                "strong": strong,
                "weak_time_s": weak,
                "weak_N": weak_N,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
