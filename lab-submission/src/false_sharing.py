"""
false_sharing.py -- Milestone 2
Reproduces cache-line false sharing with multiprocessing.Array, then
remediates it with 64-byte padding and thread-local privatization, and
computes the Karp-Flatt serial-fraction metric e(P) for each variant.

Usage: python3 src/false_sharing.py [--iterations 5000000]
"""
import argparse
import ctypes
import json
import multiprocessing as mp
import os
import time

K_DEFAULT = 5_000_000
P_VALUES = [1, 2, 4, 8]


def worker_unpadded(raw_arr, worker_id, iterations, barrier):
    # raw_arr is a lock-free RawArray view: each process writes its own
    # index, but indices 0..7 share the same 64-byte cache line, so every
    # write forces MESI invalidation of that line in every other core's L1.
    buf = raw_arr
    barrier.wait()
    for _ in range(iterations):
        buf[worker_id] += 1


def worker_padded(raw_arr, worker_id, iterations, barrier):
    # Each counter lives at index p*8 -> 8 * 8 bytes = 64 bytes apart,
    # i.e. on its own cache line.
    idx = worker_id * 8
    buf = raw_arr
    barrier.wait()
    for _ in range(iterations):
        buf[idx] += 1


def worker_privatized(raw_arr, worker_id, iterations, barrier):
    barrier.wait()
    local = 0
    for _ in range(iterations):
        local += 1
    raw_arr[worker_id] = local


def run_variant(worker_fn, P, iterations, array_len):
    # lock=False: a plain shared-memory RawArray with no multiprocessing
    # Lock/Semaphore around each access, so timing reflects hardware cache
    # coherence traffic, not Python-level mutex contention.
    shared_arr = mp.Array(ctypes.c_int64, array_len, lock=False)
    barrier = mp.Barrier(P)
    procs = [
        mp.Process(target=worker_fn, args=(shared_arr, p, iterations, barrier))
        for p in range(P)
    ]
    t0 = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    elapsed = time.perf_counter() - t0

    if worker_fn is worker_padded:
        total = sum(shared_arr[p * 8] for p in range(P))
    else:
        total = sum(shared_arr[p] for p in range(P))
    return elapsed, total


def karp_flatt(P, T1, TP):
    if P == 1:
        return None
    sp = T1 / TP
    return ((P / sp) - 1) / (P - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=K_DEFAULT)
    args = ap.parse_args()
    K = args.iterations

    os.makedirs("results", exist_ok=True)
    variants = {
        "unpadded": (worker_unpadded, 8),
        "padded": (worker_padded, 64),      # 8 counters * 8 slots spacing
        "privatized": (worker_privatized, 8),
    }

    times = {name: {} for name in variants}
    for name, (fn, arrlen) in variants.items():
        for P in P_VALUES:
            elapsed, total = run_variant(fn, P, K, arrlen)
            assert total == P * K, f"{name} P={P}: count mismatch {total} != {P*K}"
            times[name][P] = elapsed
            print(f"{name:12s} P={P:<2d} T={elapsed:.4f}s  total={total}")

    rows = []
    for name in variants:
        T1 = times[name][1]
        for P in P_VALUES:
            TP = times[name][P]
            sp = T1 / TP
            ep = karp_flatt(P, T1, TP)
            rows.append((name, P, K, TP, sp, ep))

    with open("results/false_sharing_karp_flatt.tsv", "w") as f:
        f.write("variant\tP\titerations_per_worker\ttime_s\tspeedup\tkarp_flatt_e\n")
        for name, P, K_, TP, sp, ep in rows:
            ep_s = "" if ep is None else f"{ep:.6f}"
            f.write(f"{name}\t{P}\t{K_}\t{TP:.6f}\t{sp:.4f}\t{ep_s}\n")

    print("\nWrote results/false_sharing_karp_flatt.tsv")

    summary = {
        "iterations_per_worker": K,
        "times_s": times,
        "note": (
            "unpadded e(P) should rise sharply with P (cache-line ping-pong, "
            "MESI invalidation traffic on the shared 64-byte line holding all "
            "8 int64 counters); padded/privatized e(P) should stay low and flat."
        ),
    }
    with open("results/false_sharing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
