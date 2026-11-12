
import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 230103172
READS = 2_000_000


def main():
    print("PHASE 0: Hardware and Timer Calibration")
    print("-" * 50)

    cpu_count = mp.cpu_count()

    print("Logical CPU threads:", cpu_count)

    start = time.perf_counter_ns()

    for _ in range(READS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_ns = total_ns / READS

    print("Total clock reads:", READS)
    print("Total time (ns):", total_ns)
    print("Average latency (ns):", average_ns)
    print("Total time (ms):", total_ns / 1e6)

    print("-" * 50)

    print(
        emit_verify_tag(
            "PHASE0",
            STUDENT_ID,
            average_ns,
            total_ns / 1e6
        )
    )


if __name__ == "__main__":
    main()
