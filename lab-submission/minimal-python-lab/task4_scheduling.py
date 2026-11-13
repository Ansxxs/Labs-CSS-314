
import multiprocessing as mp
import math
import statistics
import time

from harness import emit_verify_tag


STUDENT_ID = 230103172

N = 8000
PHYSICAL_CORES = 8

CHUNK_SIZES = [1, 8, 64, 512]
RUNS = 3


def calculate_item(i):
    """Calculate the skewed workload for one item."""
    k = 1 + int((i / 40) ** 2)

    x = 0.5 + (i % 100) * 0.01

    for _ in range(k):
        x = math.sin(x) + 1.0
        x = math.cos(x) + 1.0
        x = math.sqrt(abs(x) + 0.1)

    return x


def worker_block(worker_id, start, end, worker_times, start_event):
    start_event.wait()
    worker_start = time.perf_counter_ns()

    result = 0.0

    for i in range(start, end):
        result += calculate_item(i)

    worker_end = time.perf_counter_ns()

    worker_times[worker_id] = (
        worker_end - worker_start
    ) / 1_000_000_000

    return result


def worker_cyclic(worker_id, processes, worker_times, start_event):
    start_event.wait()
    worker_start = time.perf_counter_ns()

    result = 0.0

    for i in range(worker_id, N, processes):
        result += calculate_item(i)

    worker_end = time.perf_counter_ns()

    worker_times[worker_id] = (
        worker_end - worker_start
    ) / 1_000_000_000

    return result


def worker_dynamic(
    worker_id,
    counter,
    chunk_size,
    worker_times,
    start_event
):
    start_event.wait()
    worker_start = time.perf_counter_ns()

    result = 0.0

    while True:
        # Claim the next chunk using a synchronized counter.
        with counter.get_lock():
            start = counter.value
            counter.value += chunk_size

        if start >= N:
            break

        end = min(start + chunk_size, N)

        for i in range(start, end):
            result += calculate_item(i)

    worker_end = time.perf_counter_ns()

    worker_times[worker_id] = (
        worker_end - worker_start
    ) / 1_000_000_000

    return result


def run_strategy(strategy, chunk_size=None):
    worker_times = mp.Array(
        "d", PHYSICAL_CORES, lock=False
    )

    start_event = mp.Event()
    processes = []

    counter = None

    if strategy == "dynamic":
        counter = mp.Value("i", 0)

    try:
        for worker_id in range(PHYSICAL_CORES):

            if strategy == "block":
                start = N * worker_id // PHYSICAL_CORES
                end = N * (worker_id + 1) // PHYSICAL_CORES

                process = mp.Process(
                    target=worker_block,
                    args=(
                        worker_id,
                        start,
                        end,
                        worker_times,
                        start_event
                    )
                )

            elif strategy == "cyclic":
                process = mp.Process(
                    target=worker_cyclic,
                    args=(
                        worker_id,
                        PHYSICAL_CORES,
                        worker_times,
                        start_event
                    )
                )

            elif strategy == "dynamic":
                process = mp.Process(
                    target=worker_dynamic,
                    args=(
                        worker_id,
                        counter,
                        chunk_size,
                        worker_times,
                        start_event
                    )
                )

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            processes.append(process)

        # Start all workers before releasing the event.
        for process in processes:
            process.start()

        start = time.perf_counter_ns()
        start_event.set()

        for process in processes:
            process.join()

        end = time.perf_counter_ns()

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

        wall_seconds = (end - start) / 1_000_000_000

        worker_values = list(worker_times)

        average_worker_time = statistics.mean(worker_values)

        if average_worker_time > 0:
            imbalance = (
                max(worker_values) / average_worker_time
            )
        else:
            imbalance = 0.0

        return wall_seconds, imbalance, worker_values

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


def benchmark_strategy(strategy, chunk_size=None):
    wall_times = []
    imbalance_values = []
    worker_time_runs = []

    for run in range(RUNS):
        wall, imbalance, worker_values = run_strategy(
            strategy,
            chunk_size
        )

        wall_times.append(wall)
        imbalance_values.append(imbalance)
        worker_time_runs.append(worker_values)

        print(
            f"  Run {run + 1}: "
            f"Twall={wall:.3f} s, "
            f"lambda={imbalance:.3f}"
        )

    median_wall = statistics.median(wall_times)
    median_imbalance = statistics.median(imbalance_values)

    median_worker_times = []

    for worker_id in range(PHYSICAL_CORES):
        values = [
            run[worker_id]
            for run in worker_time_runs
        ]

        median_worker_times.append(
            statistics.median(values)
        )

    return (
        median_wall,
        median_imbalance,
        median_worker_times
    )


def main():
    print("=" * 85)
    print("PHASE 4: SKEWED WORKLOADS AND DYNAMIC CHUNK QUEUE")
    print("=" * 85)

    print(f"Number of items: {N}")
    print(f"Physical processes: {PHYSICAL_CORES}")
    print(f"Logical CPUs: {mp.cpu_count()}")
    print(f"Measured runs per strategy: {RUNS}")
    print()

    results = {}

    strategies = [
        ("block", None, "Contiguous Block Static"),
        ("cyclic", None, "Cyclic Round-Robin Static"),
        ("dynamic", 1, "Dynamic Queue C=1"),
        ("dynamic", 8, "Dynamic Queue C=8"),
        ("dynamic", 64, "Dynamic Queue C=64"),
        ("dynamic", 512, "Dynamic Queue C=512"),
    ]

    for strategy, chunk_size, label in strategies:
        print("\n" + "-" * 85)
        print(label)
        print("-" * 85)

        median_wall, imbalance, worker_times = (
            benchmark_strategy(strategy, chunk_size)
        )

        key = (
            strategy,
            chunk_size
        )

        results[key] = {
            "label": label,
            "wall": median_wall,
            "imbalance": imbalance,
            "worker_times": worker_times
        }

        print(f"\nMedian wall time: {median_wall:.3f} s")
        print(f"Load imbalance factor lambda: {imbalance:.3f}")

        print("Median worker active times:")

        for worker_id, worker_time in enumerate(worker_times):
            print(
                f"  Worker {worker_id}: "
                f"{worker_time:.3f} s"
            )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 85)
    print("SCHEDULING STRATEGY SUMMARY")
    print("=" * 85)

    print(
        f"{'Strategy':<30} "
        f"{'Median Twall':>15} "
        f"{'Lambda':>12}"
    )

    for result in results.values():
        print(
            f"{result['label']:<30} "
            f"{result['wall']:>12.3f} s "
            f"{result['imbalance']:>12.3f}"
        )

    # Best dynamic chunk based on median wall time.
    dynamic_results = {
        chunk: results[("dynamic", chunk)]
        for chunk in CHUNK_SIZES
    }

    best_chunk = min(
        dynamic_results,
        key=lambda chunk: dynamic_results[chunk]["wall"]
    )

    best_dynamic = dynamic_results[best_chunk]
    block_result = results[("block", None)]
    cyclic_result = results[("cyclic", None)]

    print("\nDYNAMIC QUEUE ANALYSIS")
    print(f"Best chunk size C*: {best_chunk}")
    print(f"Best dynamic Twall: {best_dynamic['wall']:.3f} s")
    print(f"Best dynamic lambda: {best_dynamic['imbalance']:.3f}")

    print("\nCOMPARISON WITH STATIC SCHEDULING")

    print(
        "Dynamic vs Block speedup: "
        f"{block_result['wall'] / best_dynamic['wall']:.3f}x"
    )

    print(
        "Dynamic vs Cyclic speedup: "
        f"{cyclic_result['wall'] / best_dynamic['wall']:.3f}x"
    )

    print("\nVERIFICATION")

    emit_verify_tag(
        "PHASE4",
        STUDENT_ID,
        block_result["wall"],
        best_dynamic["wall"]
    )

    print("\nPhase 4 completed.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
