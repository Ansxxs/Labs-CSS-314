
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import statistics
import time

from harness import emit_verify_tag


STUDENT_ID = 230103172
TOTAL_OPERATIONS = 16_000_000
STRIDES = [1, 2, 4, 8, 16, 32]
SCALING_PROCESSES = [1, 2, 4, 8, 16]
MASK = 0xFFFFFFFFFFFFFFFF
MULTIPLIER = 6364136223846793005
RUNS = 7


def worker(shm_name, slot_idx, iterations, start_event):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]
    start_event.wait()

    for i in range(iterations):
        acc = (acc * MULTIPLIER + (i | 1)) & MASK
        arr[slot_idx] = acc

    del arr
    shm.close()


def run_trial(processes, stride):
    shm = shared_memory.SharedMemory(create=True, size=4096)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    for i in range(512):
        arr[i] = i + 1

    del arr

    iterations_per_worker = TOTAL_OPERATIONS // processes
    start_event = mp.Event()
    workers = []

    try:
        for worker_id in range(processes):
            slot_idx = worker_id * stride

            process = mp.Process(
                target=worker,
                args=(
                    shm.name,
                    slot_idx,
                    iterations_per_worker,
                    start_event
                )
            )
            workers.append(process)

        start = time.perf_counter_ns()

        for process in workers:
            process.start()

        start_event.set()

        for process in workers:
            process.join()

        end = time.perf_counter_ns()

        for process in workers:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed with exit code {process.exitcode}"
                )

        return (end - start) / 1_000_000

    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join()

        shm.close()
        shm.unlink()


def benchmark(processes, stride):
    times = []

    # Two warm-up runs
    for _ in range(2):
        run_trial(processes, stride)

    for _ in range(RUNS):
        times.append(run_trial(processes, stride))

    ordered = sorted(times)
    q1 = statistics.quantiles(
        ordered, n=4, method="inclusive"
    )[0]
    q3 = statistics.quantiles(
        ordered, n=4, method="inclusive"
    )[2]

    median_ms = statistics.median(ordered)
    iqr_ms = q3 - q1
    throughput = TOTAL_OPERATIONS / (median_ms * 1000)

    return median_ms, iqr_ms, throughput


def main():
    physical_cores = 8
    logical_cores = mp.cpu_count()

    print("=" * 78)
    print("PHASE 1: FALSE SHARING AND KAR​​P-FLATT DIAGNOSTICS")
    print("=" * 78)
    print(f"Physical cores: {physical_cores}")
    print(f"Logical CPUs: {logical_cores}")
    print(f"Total operations per trial: {TOTAL_OPERATIONS:,}")
    print(f"Benchmark runs: {RUNS}")
    print()

    # ---------------------------------------------------------
    # Part A: Stride sweep
    # ---------------------------------------------------------

    print("PART A: FALSE SHARING STRIDE SWEEP")
    print("-" * 78)
    print(
        f"{'Stride':>8} {'Spacing':>10} {'Median ms':>14} "
        f"{'IQR ms':>12} {'Mops/s':>12}"
    )

    stride_results = {}

    for stride in STRIDES:
        median_ms, iqr_ms, throughput = benchmark(
            physical_cores, stride
        )

        stride_results[stride] = median_ms

        print(
            f"{stride:>8} "
            f"{stride * 8:>7} B "
            f"{median_ms:>14.3f} "
            f"{iqr_ms:>12.3f} "
            f"{throughput:>12.3f}"
        )

    # ---------------------------------------------------------
    # Part B: Process scaling and Karp-Flatt metric
    # ---------------------------------------------------------

    print("\nPART B: PROCESS SCALING")
    print("-" * 78)

    scaling_results = {}

    for stride in [1, 16]:
        print(f"\nStride = {stride} ({stride * 8} bytes)")
        print(
            f"{'Processes':>10} {'Median ms':>14} "
            f"{'Speedup':>12} {'Efficiency':>12} "
            f"{'Karp-Flatt':>12}"
        )

        scaling_results[stride] = {}

        for processes in SCALING_PROCESSES:
            median_ms, iqr_ms, throughput = benchmark(
                processes, stride
            )

            scaling_results[stride][processes] = median_ms

        baseline = scaling_results[stride][1]

        for processes in SCALING_PROCESSES:
            median_ms = scaling_results[stride][processes]
            speedup = baseline / median_ms
            efficiency = speedup / processes

            if processes == 1:
                karp_flatt = 0.0
            else:
                karp_flatt = (
                    (1 / speedup - 1 / processes)
                    / (1 - 1 / processes)
                )

            print(
                f"{processes:>10} "
                f"{median_ms:>14.3f} "
                f"{speedup:>12.3f} "
                f"{efficiency:>12.3f} "
                f"{karp_flatt:>12.5f}"
            )

    # ---------------------------------------------------------
    # Verification hash
    # ---------------------------------------------------------

    print("\nVERIFICATION")
    emit_verify_tag(
        "PHASE1",
        STUDENT_ID,
        stride_results[1],
        stride_results[16]
    )

    print("\nPhase 1 completed.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
