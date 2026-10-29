"""
stride_bench.py -- Milestone 1, Task 1.1
Cache-line stride cliff benchmark. Writes results/stride_results.tsv.

Usage: python3 src/stride_bench.py
"""
import os
import time
import numpy as np

STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]
N_ELEM = 16_777_216  # 2^24, ~134 MB of float64 -> exceeds typical L3


def touch(data, stride):
    # Slice view avoids Python-level loop overhead while still issuing
    # one real memory read per touched element at the given stride.
    view = data[::stride]
    total = 0.0
    total += view.sum()
    return total, view.shape[0]


def bench_stride(data, stride, warmup=5, reps=5):
    for _ in range(warmup):
        touch(data, stride)
    times = []
    n_touched = 0
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        _, n_touched = touch(data, stride)
        times.append(time.perf_counter_ns() - t0)
    times.sort()
    median_ns = times[len(times) // 2]
    return median_ns, n_touched


def main():
    os.makedirs("results", exist_ok=True)
    data = np.ones(N_ELEM, dtype=np.float64)

    rows = []
    print(f"{'stride':>8} {'median_ns':>12} {'elements':>12} {'ns/elem':>10}")
    for s in STRIDES:
        median_ns, n_touched = bench_stride(data, s)
        ns_per_elem = median_ns / n_touched
        rows.append((s, median_ns, n_touched, ns_per_elem))
        print(f"{s:>8} {median_ns:>12} {n_touched:>12} {ns_per_elem:>10.4f}")

    with open("results/stride_results.tsv", "w") as f:
        f.write("stride\tmedian_time_ns\telements_touched\tns_per_element\n")
        for s, t, n, nse in rows:
            f.write(f"{s}\t{t}\t{n}\t{nse:.6f}\n")

    print("\nWrote results/stride_results.tsv")


if __name__ == "__main__":
    main()
