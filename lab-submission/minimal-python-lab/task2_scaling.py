
import multiprocessing as mp
import math
import statistics
import time

from harness import emit_verify_tag


STUDENT_ID = 230103172

STRONG_N = 1_200_000
WEAK_N0 = 200_000

PROCESSES = list(range(1, 17))

WARMUP_RUNS = 1
BENCHMARK_RUNS = 7


def stage_a_worker(elements):
    """Serial Stage A: 10 sin/sqrt operations per element."""
    result = 0.0

    for i in range(elements):
        x = (i % 1000) * 0.001 + 0.5

        for _ in range(5):
            x = math.sin(x) + 1.0
            x = math.sqrt(abs(x))

        result += x

    return result


def stage_b_worker(bounds):
    """Parallel Stage B: 30 math operations per element."""
    start, end = bounds
    result = 0.0

    for i in range(start, end):
        x = (i % 1000) * 0.001 + 0.5
        y = 0.25 + (i % 17) * 0.01

        for _ in range(10):
            x = math.sin(x + y)
            y = math.cos(y + x)
            x = math.sqrt(abs(x) + 0.5)

        result += x + y

    return result


def make_partitions(total_elements, processes):
    partitions = []

    for worker_id in range(processes):
        start = total_elements * worker_id // processes
        end = total_elements * (worker_id + 1) // processes
        partitions.append((start, end))

    return partitions


def run_pipeline(pool, processes, total_elements):
    stage_a_elements = int(total_elements * 0.05)

    start = time.perf_counter_ns()

    # Stage A: serial work in the parent process
    stage_a_start = time.perf_counter_ns()
    result_a = stage_a_worker(stage_a_elements)
    stage_a_end = time.perf_counter_ns()

    stage_a_ms = (stage_a_end - stage_a_start) / 1_000_000

    # Stage B: parallel work using a persistent process pool
    partitions = make_partitions(total_elements, processes)
    results = pool.map(stage_b_worker, partitions)

    end = time.perf_counter_ns()

    # Consume results to ensure the computations are retained
    checksum = result_a + sum(results)

    total_ms = (end - start) / 1_000_000

    return total_ms, stage_a_ms, checksum


def benchmark_pipeline(processes, total_elements):
    stage_a_times = []
    total_times = []

    with mp.Pool(processes=processes) as pool:

        # Warm-up
        for _ in range(WARMUP_RUNS):
            run_pipeline(pool, processes, total_elements)

        # Measured runs
        for _ in range(BENCHMARK_RUNS):
            total_ms, stage_a_ms, checksum = run_pipeline(
                pool,
                processes,
                total_elements
            )

            total_times.append(total_ms)
            stage_a_times.append(stage_a_ms)

    return (
        statistics.median(total_times),
        statistics.median(stage_a_times)
    )


def main():
    logical_cpus = mp.cpu_count()

    print("=" * 85)
    print("PHASE 2: STRONG SCALING VS WEAK SCALING")
    print("=" * 85)
    print(f"Logical CPUs: {logical_cpus}")
    print(f"Strong scaling N: {STRONG_N:,}")
    print(f"Weak scaling N0: {WEAK_N0:,}")
    print(f"Measured runs: {BENCHMARK_RUNS}")
    print()

    processes_list = list(range(1, logical_cpus + 1))

    # ---------------------------------------------------------
    # PART A: STRONG SCALING
    # ---------------------------------------------------------

    print("PART A: STRONG SCALING")
    print("-" * 85)

    strong_results = {}

    print(
        f"{'Processes':>10} "
        f"{'N':>12} "
        f"{'Stage A ms':>14} "
        f"{'Total ms':>14} "
        f"{'Speedup':>12} "
        f"{'Efficiency':>12}"
    )

    for processes in processes_list:
        total_ms, stage_a_ms = benchmark_pipeline(
            processes,
            STRONG_N
        )

        strong_results[processes] = {
            "total_ms": total_ms,
            "stage_a_ms": stage_a_ms
        }

        baseline = strong_results[1]["total_ms"]
        speedup = baseline / total_ms
        efficiency = speedup / processes

        print(
            f"{processes:>10} "
            f"{STRONG_N:>12,} "
            f"{stage_a_ms:>14.3f} "
            f"{total_ms:>14.3f} "
            f"{speedup:>12.3f} "
            f"{efficiency:>12.3f}"
        )

    # Amdahl's law
    stage_a_fraction = (
        strong_results[1]["stage_a_ms"]
        / strong_results[1]["total_ms"]
    )

    amdahl_limit = 1.0 / stage_a_fraction

    strong_speedup_max = (
        strong_results[logical_cpus]["total_ms"]
    )

    strong_speedup_max = (
        strong_results[1]["total_ms"] / strong_speedup_max
    )

    print("\nAMDAHL'S LAW")
    print(f"Serial fraction: {stage_a_fraction:.6f}")
    print(f"Theoretical speedup limit: {amdahl_limit:.3f}")
    print(
        f"Measured speedup at {logical_cpus} processes: "
        f"{strong_speedup_max:.3f}"
    )

    # ---------------------------------------------------------
    # PART B: WEAK SCALING
    # ---------------------------------------------------------

    print("\nPART B: WEAK SCALING")
    print("-" * 85)

    weak_results = {}

    print(
        f"{'Processes':>10} "
        f"{'Total N':>12} "
        f"{'Stage A ms':>14} "
        f"{'Total ms':>14} "
        f"{'Efficiency':>12} "
        f"{'Gustafson':>12}"
    )

    for processes in processes_list:
        total_elements = processes * WEAK_N0

        total_ms, stage_a_ms = benchmark_pipeline(
            processes,
            total_elements
        )

        weak_results[processes] = {
            "total_ms": total_ms,
            "stage_a_ms": stage_a_ms
        }

        if processes == 1:
            weak_efficiency = 1.0
            gustafson_speedup = 1.0
        else:
            weak_efficiency = (
                weak_results[1]["total_ms"] / total_ms
            )

            alpha = stage_a_ms / total_ms

            gustafson_speedup = (
                processes - alpha * (processes - 1)
            )

        print(
            f"{processes:>10} "
            f"{total_elements:>12,} "
            f"{stage_a_ms:>14.3f} "
            f"{total_ms:>14.3f} "
            f"{weak_efficiency:>12.3f} "
            f"{gustafson_speedup:>12.3f}"
        )

    # ---------------------------------------------------------
    # PART C: ARCHITECTURAL INFLECTION
    # ---------------------------------------------------------

    print("\nPART C: ARCHITECTURAL INFLECTION")
    print("-" * 85)

    if logical_cpus > 8:
        t8 = strong_results[8]["total_ms"]
        t9 = strong_results[9]["total_ms"]

        speedup_8 = strong_results[1]["total_ms"] / t8
        speedup_9 = strong_results[1]["total_ms"] / t9

        marginal_gain = speedup_9 - speedup_8

        print(f"Speedup at 8 processes: {speedup_8:.3f}")
        print(f"Speedup at 9 processes: {speedup_9:.3f}")
        print(f"Marginal speedup change (9 - 8): {marginal_gain:.4f}")

        if marginal_gain > 0:
            print("Adding the 9th process improved performance.")
        else:
            print("Adding the 9th process did not improve performance.")
    else:
        print("Not enough logical CPUs to compare 8 and 9 processes.")

    # ---------------------------------------------------------
    # VERIFICATION
    # ---------------------------------------------------------

    final_strong_speedup = (
        strong_results[logical_cpus]["total_ms"]
    )
    final_strong_speedup = (
        strong_results[1]["total_ms"] / final_strong_speedup
    )

    final_weak_gustafson = (
        logical_cpus
        - (
            weak_results[logical_cpus]["stage_a_ms"]
            / weak_results[logical_cpus]["total_ms"]
        ) * (logical_cpus - 1)
    )

    print("\nVERIFICATION")
    emit_verify_tag(
        "PHASE2",
        STUDENT_ID,
        final_strong_speedup,
        final_weak_gustafson
    )

    print("\nPhase 2 completed.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
