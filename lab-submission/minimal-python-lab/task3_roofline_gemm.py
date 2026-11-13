
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import statistics
import time
import math

from harness import emit_verify_tag


STUDENT_ID = 230103172
PHYSICAL_CORES = 8

BUFFER_SIZE = 64 * 1024 * 1024
GEMM_N = 256

RUNS = 7
WARMUP = 1

TILE_SIZES = [8, 16, 32, 64, 96, 128]

# ---------------------------------------------------------
# Shared globals for worker processes
# ---------------------------------------------------------

_MEM_SRC = None
_MEM_DST = None
_MEM_SRC_VIEW = None
_MEM_DST_VIEW = None

_GEMM_A_SHM = None
_GEMM_B_SHM = None
_GEMM_C_SHM = None

_GEMM_A = None
_GEMM_B = None
_GEMM_C = None
_GEMM_N_WORKER = 0


# ---------------------------------------------------------
# General benchmark helpers
# ---------------------------------------------------------

def median_iqr(values):
    ordered = sorted(values)

    q1 = statistics.quantiles(
        ordered, n=4, method="inclusive"
    )[0]

    q3 = statistics.quantiles(
        ordered, n=4, method="inclusive"
    )[2]

    return statistics.median(ordered), q3 - q1


# ---------------------------------------------------------
# PART A: MEMORY BANDWIDTH
# ---------------------------------------------------------

def init_memory_worker(src_name, dst_name):
    global _MEM_SRC, _MEM_DST
    global _MEM_SRC_VIEW, _MEM_DST_VIEW

    _MEM_SRC = shared_memory.SharedMemory(name=src_name)
    _MEM_DST = shared_memory.SharedMemory(name=dst_name)

    _MEM_SRC_VIEW = _MEM_SRC.buf
    _MEM_DST_VIEW = _MEM_DST.buf


def copy_memory_chunk(bounds):
    start, end = bounds

    _MEM_DST_VIEW[start:end] = _MEM_SRC_VIEW[start:end]

    return end - start


def measure_memory_bandwidth():
    print("\nPART 3A-1: SUSTAINED MEMORY BANDWIDTH")
    print("-" * 70)

    src = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )

    dst = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )

    src.buf[:] = b"\x5A" * BUFFER_SIZE

    chunks = []

    for worker_id in range(PHYSICAL_CORES):
        start = BUFFER_SIZE * worker_id // PHYSICAL_CORES
        end = BUFFER_SIZE * (worker_id + 1) // PHYSICAL_CORES
        chunks.append((start, end))

    times = []

    try:
        with mp.Pool(
            processes=PHYSICAL_CORES,
            initializer=init_memory_worker,
            initargs=(src.name, dst.name)
        ) as pool:

            for _ in range(WARMUP):
                pool.map(copy_memory_chunk, chunks)

            for run in range(RUNS):
                start = time.perf_counter_ns()

                copied = pool.map(copy_memory_chunk, chunks)

                end = time.perf_counter_ns()

                elapsed = (end - start) / 1_000_000_000
                times.append(elapsed)

                print(
                    f"Run {run + 1}: "
                    f"{elapsed * 1000:.3f} ms, "
                    f"{sum(copied) / 1_000_000:.1f} MB copied"
                )

        median_seconds, iqr_seconds = median_iqr(times)

        # Read + write traffic: 64 MB read and 64 MB written.
        bandwidth_gbs = (
            2 * 64 * 1_000_000
            / median_seconds
            / 1_000_000_000
        )

        print(f"\nMedian time: {median_seconds * 1000:.3f} ms")
        print(f"IQR: {iqr_seconds * 1000:.3f} ms")
        print(f"Sustained bandwidth: {bandwidth_gbs:.3f} GB/s")

        return bandwidth_gbs

    finally:
        src.close()
        dst.close()
        src.unlink()
        dst.unlink()


# ---------------------------------------------------------
# PART A: COMPUTE CEILING
# ---------------------------------------------------------

def compute_worker(iterations):
    factor = 1.0000001

    a0 = 0.1
    a1 = 0.2
    a2 = 0.3
    a3 = 0.4
    a4 = 0.5
    a5 = 0.6
    a6 = 0.7
    a7 = 0.8
    a8 = 0.9
    a9 = 1.0
    a10 = 1.1
    a11 = 1.2
    a12 = 1.3
    a13 = 1.4
    a14 = 1.5
    a15 = 1.6

    for _ in range(iterations):
        a0 = a0 * factor + 0.1
        a1 = a1 * factor + 0.1
        a2 = a2 * factor + 0.1
        a3 = a3 * factor + 0.1
        a4 = a4 * factor + 0.1
        a5 = a5 * factor + 0.1
        a6 = a6 * factor + 0.1
        a7 = a7 * factor + 0.1
        a8 = a8 * factor + 0.1
        a9 = a9 * factor + 0.1
        a10 = a10 * factor + 0.1
        a11 = a11 * factor + 0.1
        a12 = a12 * factor + 0.1
        a13 = a13 * factor + 0.1
        a14 = a14 * factor + 0.1
        a15 = a15 * factor + 0.1

    return (
        a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
        + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15
    )


def measure_compute_ceiling(bandwidth_gbs):
    print("\nPART 3A-2: COMPUTE CEILING")
    print("-" * 70)

    # 16 independent accumulators.
    # Each accumulator performs 2 FLOPs per iteration.
    # 1,875,000 * 16 * 2 = 60,000,000 FLOPs per worker.
    iterations = 1_875_000
    flops_per_worker = 60_000_000

    tasks = [iterations] * PHYSICAL_CORES
    times = []

    with mp.Pool(processes=PHYSICAL_CORES) as pool:

        for _ in range(WARMUP):
            pool.map(compute_worker, tasks)

        for run in range(RUNS):
            start = time.perf_counter_ns()

            results = pool.map(compute_worker, tasks)

            end = time.perf_counter_ns()

            elapsed = (end - start) / 1_000_000_000
            times.append(elapsed)

            print(
                f"Run {run + 1}: {elapsed:.3f} s, "
                f"checksum={sum(results):.3f}"
            )

    median_seconds, iqr_seconds = median_iqr(times)

    total_flops = flops_per_worker * PHYSICAL_CORES

    p_sustained = (
        total_flops / median_seconds / 1_000_000_000
    )

    ridge_point = p_sustained / bandwidth_gbs

    print(f"\nMedian time: {median_seconds:.3f} s")
    print(f"IQR: {iqr_seconds:.3f} s")
    print(f"Total nominal FLOPs: {total_flops:,}")
    print(f"Sustained compute: {p_sustained:.3f} GFLOP/s")
    print(f"Roofline ridge point: {ridge_point:.3f} FLOP/byte")

    return p_sustained, ridge_point


# ---------------------------------------------------------
# PART B: GEMM INITIALIZATION
# ---------------------------------------------------------

def init_gemm_worker(a_name, b_name, c_name, n):
    global _GEMM_A_SHM, _GEMM_B_SHM, _GEMM_C_SHM
    global _GEMM_A, _GEMM_B, _GEMM_C, _GEMM_N_WORKER

    _GEMM_A_SHM = shared_memory.SharedMemory(name=a_name)
    _GEMM_B_SHM = shared_memory.SharedMemory(name=b_name)
    _GEMM_C_SHM = shared_memory.SharedMemory(name=c_name)

    count = n * n

    _GEMM_A = (ctypes.c_double * count).from_buffer(_GEMM_A_SHM.buf)
    _GEMM_B = (ctypes.c_double * count).from_buffer(_GEMM_B_SHM.buf)
    _GEMM_C = (ctypes.c_double * count).from_buffer(_GEMM_C_SHM.buf)

    _GEMM_N_WORKER = n


def gemm_naive(bounds):
    """Variant 1: i-j-k, strided access to B."""
    row_start, row_end = bounds
    n = _GEMM_N_WORKER

    for i in range(row_start, row_end):
        for j in range(n):
            total = 0.0

            for k in range(n):
                total += (
                    _GEMM_A[i * n + k]
                    * _GEMM_B[k * n + j]
                )

            _GEMM_C[i * n + j] = total


def gemm_reordered(bounds):
    """Variant 2: i-k-j, unit-stride access to B and C."""
    row_start, row_end = bounds
    n = _GEMM_N_WORKER

    for i in range(row_start, row_end):
        row_offset = i * n

        for k in range(n):
            a_ik = _GEMM_A[row_offset + k]
            b_offset = k * n

            for j in range(n):
                _GEMM_C[row_offset + j] += (
                    a_ik * _GEMM_B[b_offset + j]
                )


def gemm_blocked(bounds, block_size):
    """Variant 3: six-loop cache-blocked i-k-j GEMM."""
    row_start, row_end = bounds
    n = _GEMM_N_WORKER
    bs = block_size

    for ii in range(row_start, row_end, bs):
        i_end = min(ii + bs, row_end)

        for kk in range(0, n, bs):
            k_end = min(kk + bs, n)

            for jj in range(0, n, bs):
                j_end = min(jj + bs, n)

                for i in range(ii, i_end):
                    row_offset = i * n

                    for k in range(kk, k_end):
                        a_ik = _GEMM_A[row_offset + k]
                        b_offset = k * n

                        for j in range(jj, j_end):
                            _GEMM_C[row_offset + j] += (
                                a_ik * _GEMM_B[b_offset + j]
                            )


def reset_matrix_c(c_array, count):
    for i in range(count):
        c_array[i] = 0.0


def max_difference(reference, actual):
    return max(
        abs(reference[i] - actual[i])
        for i in range(len(reference))
    )


def benchmark_gemm(pool, c_array, count, chunks, function, extra=None):
    times = []

    for run in range(WARMUP + RUNS):
        reset_matrix_c(c_array, count)

        start = time.perf_counter_ns()

        if extra is None:
            pool.map(function, chunks)
        else:
            pool.starmap(
                function,
                [(chunk, extra) for chunk in chunks]
            )

        end = time.perf_counter_ns()

        if run >= WARMUP:
            times.append((end - start) / 1_000_000_000)

    median_seconds, iqr_seconds = median_iqr(times)

    return median_seconds, iqr_seconds


# ---------------------------------------------------------
# PART B: GEMM BENCHMARK
# ---------------------------------------------------------

def measure_gemm():
    print("\nPART 3B: GEMM AND CACHE-BLOCKED TILE SWEEP")
    print("-" * 85)

    n = GEMM_N
    count = n * n
    buffer_bytes = count * ctypes.sizeof(ctypes.c_double)

    shm_a = shared_memory.SharedMemory(
        create=True, size=buffer_bytes
    )
    shm_b = shared_memory.SharedMemory(
        create=True, size=buffer_bytes
    )
    shm_c = shared_memory.SharedMemory(
        create=True, size=buffer_bytes
    )

    a = (ctypes.c_double * count).from_buffer(shm_a.buf)
    b = (ctypes.c_double * count).from_buffer(shm_b.buf)
    c = (ctypes.c_double * count).from_buffer(shm_c.buf)

    for i in range(n):
        for j in range(n):
            a[i * n + j] = ((i + j) % 17) * 0.01 + 0.1
            b[i * n + j] = ((i * 2 + j) % 13) * 0.02 + 0.2

    chunks = []

    for worker_id in range(PHYSICAL_CORES):
        row_start = n * worker_id // PHYSICAL_CORES
        row_end = n * (worker_id + 1) // PHYSICAL_CORES
        chunks.append((row_start, row_end))

    results = {}

    try:
        with mp.Pool(
            processes=PHYSICAL_CORES,
            initializer=init_gemm_worker,
            initargs=(shm_a.name, shm_b.name, shm_c.name, n)
        ) as pool:

            # Variant 1: i-j-k
            print("\nVariant 1: Naive i-j-k")

            t1, iqr1 = benchmark_gemm(
                pool, c, count, chunks, gemm_naive
            )

            gflops1 = (2 * n ** 3) / (t1 * 1_000_000_000)

            results["variant1"] = {
                "time": t1,
                "iqr": iqr1,
                "gflops": gflops1
            }

            print(
                f"Median: {t1:.4f} s | "
                f"IQR: {iqr1:.4f} s | "
                f"Performance: {gflops1:.4f} GFLOP/s"
            )

            # Variant 2: i-k-j
            print("\nVariant 2: Reordered i-k-j")

            t2, iqr2 = benchmark_gemm(
                pool, c, count, chunks, gemm_reordered
            )

            reference = list(c)

            gflops2 = (2 * n ** 3) / (t2 * 1_000_000_000)

            results["variant2"] = {
                "time": t2,
                "iqr": iqr2,
                "gflops": gflops2
            }

            print(
                f"Median: {t2:.4f} s | "
                f"IQR: {iqr2:.4f} s | "
                f"Performance: {gflops2:.4f} GFLOP/s"
            )

            # Variant 3: blocked GEMM
            print("\nVariant 3: Cache-blocked i-k-j")
            print("-" * 85)

            print(
                f"{'Block size':>12} "
                f"{'Tile footprint':>18} "
                f"{'Median s':>12} "
                f"{'IQR s':>12} "
                f"{'GFLOP/s':>12} "
                f"{'Intensity':>12}"
            )

            best_block = None
            best_gflops = -1.0
            best_matrix = None

            for bs in TILE_SIZES:
                t, iqr = benchmark_gemm(
                    pool,
                    c,
                    count,
                    chunks,
                    gemm_blocked,
                    bs
                )

                gflops = (2 * n ** 3) / (
                    t * 1_000_000_000
                )

                tile_bytes = 3 * bs * bs * 8
                tile_kb = tile_bytes / 1024

                intensity = bs / 12

                print(
                    f"{bs:>12} "
                    f"{tile_kb:>15.2f} KB "
                    f"{t:>12.4f} "
                    f"{iqr:>12.4f} "
                    f"{gflops:>12.4f} "
                    f"{intensity:>12.3f}"
                )

                results[bs] = {
                    "time": t,
                    "iqr": iqr,
                    "gflops": gflops,
                    "tile_kb": tile_kb,
                    "intensity": intensity
                }

                if gflops > best_gflops:
                    best_gflops = gflops
                    best_block = bs
                    best_matrix = list(c)

            difference = max_difference(reference, best_matrix)

            print("\nNUMERICAL VERIFICATION")
            print(f"Best block size: {best_block}")
            print(f"Max |C(blocked) - C(i-k-j)|: {difference:.12e}")

            if difference < 1e-9:
                print("Verification PASSED.")
            else:
                print("Verification FAILED.")

            print("\nGEMM SUMMARY")
            print(f"Variant 1: {gflops1:.4f} GFLOP/s")
            print(f"Variant 2: {gflops2:.4f} GFLOP/s")
            print(f"Best blocked: {best_gflops:.4f} GFLOP/s")
            print(f"Best tile size: {best_block}")

            return gflops1, best_gflops, best_block, difference

    finally:
        del a
        del b
        del c

        shm_a.close()
        shm_b.close()
        shm_c.close()

        shm_a.unlink()
        shm_b.unlink()
        shm_c.unlink()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("=" * 85)
    print("PHASE 3: EMPIRICAL ROOFLINE AND CACHE-BLOCKED GEMM")
    print("=" * 85)

    print(f"Physical processes: {PHYSICAL_CORES}")
    print(f"Logical CPUs: {mp.cpu_count()}")
    print(f"Memory buffer size: {BUFFER_SIZE / 1_000_000:.1f} MB")
    print(f"GEMM matrix size: {GEMM_N} x {GEMM_N}")

    bandwidth = measure_memory_bandwidth()

    compute_gflops, ridge_point = measure_compute_ceiling(
        bandwidth
    )

    variant1_gflops, blocked_gflops, best_block, difference = (
        measure_gemm()
    )

    print("\nROOFLINE SUMMARY")
    print("-" * 70)
    print(f"Sustained bandwidth: {bandwidth:.4f} GB/s")
    print(f"Compute ceiling: {compute_gflops:.4f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.4f} FLOP/byte")

    print("\nGEMM SUMMARY")
    print(f"Variant 1 GFLOP/s: {variant1_gflops:.4f}")
    print(f"Best blocked GFLOP/s: {blocked_gflops:.4f}")
    print(f"Best block size: {best_block}")
    print(f"Numerical difference: {difference:.12e}")

    print("\nVERIFICATION")

    emit_verify_tag(
        "PHASE3",
        STUDENT_ID,
        variant1_gflops,
        blocked_gflops
    )

    print("\nPhase 3 completed.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
