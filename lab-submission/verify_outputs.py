"""
verify_outputs.py -- Milestone 4, Task 4.1
Checks the results produced by Milestones 0-3, computes a salted SHA-256
digest of the numeric outputs, and writes results/verification_receipt.sig.

Usage: python3 verify_outputs.py --student-id <YOUR_ID>
"""
import argparse
import hashlib
import json
import os
import sys


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_tsv(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        lines = [l.rstrip("\n").split("\t") for l in f]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True)
    args = ap.parse_args()
    ok = True
    report = []

    # Milestone 0
    host = load("results/host_profile.json")
    salted = load("results/salted_params.json")
    if host and salted:
        report.append(("PASS", "Milestone 0: Host topology and salted parameters present."))
    else:
        ok = False
        report.append(("FAIL", "Milestone 0: missing host_profile.json or salted_params.json."))

    # Milestone 1
    interchange = load("results/loop_interchange.json")
    stride = load_tsv("results/stride_results.tsv")
    if interchange and stride:
        consistent = interchange.get("numerically_consistent", False)
        digest = hashlib.sha256(
            f"{args.student_id}:{round(interchange['median_time_interchanged_ikj_s'], 6)}".encode()
        ).hexdigest()[:8]
        if consistent:
            report.append(("PASS", f"Milestone 1: interchanged-loop result matches naive baseline (digest {digest}...)."))
        else:
            ok = False
            report.append(("FAIL", f"Milestone 1: numerical mismatch, max_abs_diff={interchange.get('max_abs_diff')}."))
    else:
        ok = False
        report.append(("FAIL", "Milestone 1: missing stride_results.tsv or loop_interchange.json."))

    # Milestone 2
    fs_rows = load_tsv("results/false_sharing_karp_flatt.tsv")
    if fs_rows and len(fs_rows) > 1:
        header, data = fs_rows[0], fs_rows[1:]
        ok2 = True
        for row in data:
            if len(row) < 3:
                ok2 = False
                break
        if ok2:
            report.append(("PASS", "Milestone 2: unpadded/padded/privatized counter runs present and well-formed."))
        else:
            ok = False
            ok2 = False
            report.append(("FAIL", "Milestone 2: malformed rows in false_sharing_karp_flatt.tsv."))
    else:
        ok = False
        report.append(("FAIL", "Milestone 2: missing or empty false_sharing_karp_flatt.tsv."))

    # Milestone 3
    scaling = load_tsv("results/scaling_metrics.tsv")
    roofline = load("results/roofline_summary.json")
    if scaling and roofline:
        report.append(("PASS", "Milestone 3: strong/weak scaling and roofline classification present."))
    else:
        ok = False
        report.append(("FAIL", "Milestone 3: missing scaling_metrics.tsv or roofline_summary.json."))

    for status, msg in report:
        print(f"[{status}] {msg}")

    if ok:
        payload = args.student_id.encode() + b"|" + json.dumps(report, sort_keys=True).encode()
        sig = hashlib.sha256(payload).hexdigest()
        with open("results/verification_receipt.sig", "w") as f:
            f.write(sig + "\n")
        print("\nAll empirical datasets verified successfully. Emitting verification_receipt.sig.")
        print(f"Signature: {sig}")
    else:
        print("\nVerification FAILED -- fix the missing/incorrect milestones above before committing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
