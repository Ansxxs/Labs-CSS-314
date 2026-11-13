
from dataclasses import dataclass
import multiprocessing as mp
import statistics
import time
from typing import Callable


@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float


def fnv1a_64(data: str) -> str:
    offset_basis = 14695981039346656037
    prime = 1099511628211
    mask = 0xFFFFFFFFFFFFFFFF

    h = offset_basis
    for byte in data.encode("utf-8"):
        h ^= byte
        h = (h * prime) & mask

    return f"{h:016X}"


def run_benchmark(
    func: Callable,
    warmup: int = 2,
    runs: int = 7
) -> BenchStats:
    for _ in range(warmup):
        func()

    times_ms = []

    for _ in range(runs):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times_ms.append((end - start) / 1_000_000)

    sorted_times = sorted(times_ms)
    q1 = statistics.quantiles(sorted_times, n=4, method="inclusive")[0]
    q3 = statistics.quantiles(sorted_times, n=4, method="inclusive")[2]

    return BenchStats(
        min_ms=min(sorted_times),
        median_ms=statistics.median(sorted_times),
        iqr_ms=q3 - q1
    )


def emit_verify_tag(phase: str, seed: int, m1: float, m2: float) -> str:
    hw = mp.cpu_count()
    payload = f"{phase}|{seed}|{hw}|{m1:.4f}|{m2:.4f}"
    tag = fnv1a_64(payload)

    result = f"VERIFY_HASH[{phase}]={tag}"
    print(result)
    return result
