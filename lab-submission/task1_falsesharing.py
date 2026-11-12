
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103172
M = 24_000_000
STRIDES = [1, 2, 4, 8, 16, 32]


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        acc = arr[slot_idx]
        mask = 0xFFFFFFFFFFFFFFFF
        mult = 6364136223846793005

        for i in range(iters):
            acc = (acc * mult + (i | 1)) & mask
            arr[slot_idx] = acc

        del arr

    finally:
        shm.close()


def run_trial(shm_name, p, stride):
    iters = M // p
    processes = []

    for t in range(p):
        slot_idx = t * stride

        process = mp.Process(
            target=fs_worker,
            args=(shm_name, slot_idx, iters)
        )

        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker failed: exit code {process.exitcode}"
            )


def benchmark_config(p, stride):
    shm = shared_memory.SharedMemory(
        create=True,
        size=4096
    )

    try:
        fn = lambda: run_trial(shm.name, p, stride)

        stats = run_benchmark(
            fn,
            warmup_runs=2,
            timed_runs=7
        )

        throughput = M * 1000 / stats.median_ms

        return stats, throughput

    finally:
        shm.close()
        shm.unlink()


def print_stride_sweep():
    p = 8

    print("\nTABLE 1A: STRIDE SENSITIVITY")
    print("Processes:", p)
    print("-" * 75)
    print(
        f"{'Stride':>8} {'Bytes':>8} "
        f"{'Median ms':>14} {'IQR ms':>12} "
        f"{'Mops/s':>12}"
    )

    results = {}

    for stride in STRIDES:
        print(f"Testing stride={stride}...", flush=True)

        stats, throughput = benchmark_config(p, stride)

        results[stride] = stats.median_ms

        print(
            f"{stride:>8} {stride * 8:>8} "
            f"{stats.median_ms:>14.3f} "
            f"{stats.iqr_ms:>12.3f} "
            f"{throughput:>12.3f}"
        )

    return results


def print_scaling_sweep(stride):
    process_counts = [1, 2, 4, 8, 16]

    print(f"\nTABLE 1B: PROCESS SCALING, STRIDE={stride}")
    print("-" * 90)

    print(
        f"{'p':>5} {'Median ms':>14} "
        f"{'Speedup':>12} {'Efficiency':>12} "
        f"{'Karp-Flatt':>14}"
    )

    results = {}

    for p in process_counts:
        print(f"Testing p={p}, stride={stride}...", flush=True)

        stats, throughput = benchmark_config(p, stride)
        results[p] = stats.median_ms

    t1 = results[1]

    for p in process_counts:
        tp = results[p]
        speedup = t1 / tp
        efficiency = speedup / p

        if p == 1:
            karp_flatt = float("nan")
        else:
            karp_flatt = (
                (1 / speedup - 1 / p) /
                (1 - 1 / p)
            )

        print(
            f"{p:>5} {tp:>14.3f} "
            f"{speedup:>12.4f} "
            f"{efficiency:>12.4f} "
            f"{karp_flatt:>14.6f}"
        )

    return results


def main():
    print("=" * 75)
    print("PHASE 1: FALSE SHARING")
    print("CPU physical cores: 8")
    print("Total operations:", M)
    print("=" * 75)

    stride_results = print_stride_sweep()

    print("\nPROCESS SCALING EXPERIMENTS")

    results_stride1 = print_scaling_sweep(1)
    results_stride16 = print_scaling_sweep(16)

    print("\nVERIFY HASH")

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            stride_results[1],
            stride_results[16]
        )
    )

    print("\nPHASE 1 COMPLETED")


if __name__ == "__main__":
    mp.freeze_support()
    main()
