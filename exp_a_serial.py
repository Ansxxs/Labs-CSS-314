"""
exp_a_serial.py -- Part 1: The Stubborn Serial Tail

Compute work is parallelized cleanly across worker processes, but the
master process executes an intentional 0.40s sequential pause before
dispatching tasks. This models a fixed Amdahl serial fraction: as P
grows, e(p) should stay roughly flat around ~0.10-0.15.
"""
import time
import math
from multiprocessing import Pool


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    t0 = time.perf_counter()
    time.sleep(0.40)  # Intentional sequential bottleneck (~12% serial fraction)
    with Pool(processes=workers) as pool:
        pool.map(heavy_math, [chunk_size] * total_chunks)
    return time.perf_counter() - t0


if __name__ == "__main__":
    cores = [1, 2, 4]
    runtimes = {p: run_experiment(p) for p in cores}

    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s | Speedup=1.00x")
    for p in [2, 4]:
        tp = runtimes[p]
        sp = t1 / tp
        e = ((p * tp / t1) - 1.0) / (p - 1.0)
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | Karp-Flatt e={e:.3f}")
