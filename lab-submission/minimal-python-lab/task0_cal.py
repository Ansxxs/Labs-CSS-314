
import multiprocessing as mp
import platform
import time

from harness import emit_verify_tag


STUDENT_ID = 230103172
ITERATIONS = 1_000_000


def main():
    print("=" * 65)
    print("PHASE 0: HARDWARE TOPOLOGY AND TIMER CALIBRATION")
    print("=" * 65)

    print(f"CPU: {platform.processor()}")
    print(f"Logical CPUs: {mp.cpu_count()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Timer iterations: {ITERATIONS:,}")

    start = time.perf_counter_ns()

    for _ in range(ITERATIONS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_latency_ns = total_ns / ITERATIONS
    total_ms = total_ns / 1_000_000

    print("\nTimer calibration results:")
    print(f"Total elapsed time: {total_ms:.4f} ms")
    print(f"Average timer call latency: {average_latency_ns:.4f} ns")

    emit_verify_tag(
        "PHASE0",
        STUDENT_ID,
        average_latency_ns,
        total_ms
    )


if __name__ == "__main__":
    main()
