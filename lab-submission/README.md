# CS482 Lab Practicum — Submission

**Student ID: 230103172**

```bash
pip install numpy psutil           # numba optional, speeds up Milestone 1
python3 src/profile_env.py --student-id <YOUR_ID>
git add . && git commit -m "M0: Environment and salting configuration" && git tag M0-ENV

python3 src/stride_bench.py
python3 src/dense_kernel.py --student-id <YOUR_ID>
git add . && git commit -m "M1: Memory stride and loop interchange benchmarking" && git tag M1-LOCALITY

python3 src/false_sharing.py
git add . && git commit -m "M2: False sharing reproduction and Karp-Flatt analysis" && git tag M2-SHARING

python3 src/scaling_study.py --student-id <YOUR_ID>
git add . && git commit -m "M3: Strong and weak scaling evaluation" && git tag M3-SCALING

python3 verify_outputs.py --student-id <YOUR_ID>
git add . && git commit -m "M4: Final verification and completed practicum submission" && git tag M4-FINAL
git push origin main --tags
```


