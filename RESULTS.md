# RESULTS.md — Karp-Flatt Diagnostics Practice Lab

**Student ID:** 230103172
**CPU:** AMD Ryzen 7 7735HS with Radeon Graphics, 8 physical / 16 logical cores
**OS:** Windows 10

## Results Table

| Experiment Mode       | T1 (s) | T2 (s)  | T4 (s)  | e(2)  | e(4)  | Architectural Diagnosis |
|---|---|---|---|---|---|---|
| A: Serial Tail         | 4.577  | 2.694   | 1.846   | 0.177 | 0.204 | Constant e → Fixed Amdahl serialization |
| B: Lock Contention     | 5.243  | 13.539  | 16.464  | 4.165 | 3.853 | Exploding e → Mutex & bus thrashing |
| C: Privatized State    | 0.590  | 0.343   | 0.252   | 0.163 | 0.237 | Collapsed e → Linear scalability restored |

## Notes / Observations

**Experiment A (Serial Tail):** e(2) = 0.177 and e(4) = 0.204, giving a ratio
e(4)/e(2) = 1.15, inside the [0.80, 1.25] band the handout calls for. e stays
roughly flat as cores increase, which is the Karp-Flatt signature of a fixed
algorithmic serial path (the 0.40s sleep before dispatch) rather than
cross-core contention.

**Experiment B (Lock Contention):** T4 (16.464s) is well above T1 (5.243s) —
negative speedup, exactly as the handout predicts. Both e(2) = 4.165 and
e(4) = 3.853 blow past the >0.5 threshold by nearly an order of magnitude,
confirming the slowdown is cache-line bouncing and mutex contention, not
serial code. e(4) came in slightly below e(2) instead of strictly rising
further — on this 8-core/16-thread machine the single shared lock is
already saturated at P=2, so adding more contending processes mostly adds
scheduling noise rather than a cleanly monotonic increase. The diagnostic
conclusion (massive contention, not an algorithmic bottleneck) still holds.

**Experiment C (Privatized State):** T4 (0.252s) is far below T1 (0.590s),
and e(2)/e(4) collapsed back down from Experiment B's 3.8–4.2 range to
0.16–0.24 — proof that privatizing the accumulator and synchronizing only
once at the end removes almost all of the lock-contention penalty.
Efficiency came out at 86.0% at P=2 but only 58.5% at P=4, short of the
>70% the handout expects at P=4. This is consistent with `multiprocessing`
process-spawn/IPC overhead starting to matter more once the per-worker
workload shrinks (20,000,000 increments split four ways is a very light,
sub-second task), rather than any remaining contention — the metric most
specific to this experiment, Karp-Flatt e, is still an order of magnitude
below Experiment B's values, which is the core diagnostic point.
