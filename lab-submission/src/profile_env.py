"""
profile_env.py -- Milestone 0, Task 0.1
Introspects the host platform and writes results/host_profile.json and
results/salted_params.json.

Usage:
    python3 src/profile_env.py --student-id <YOUR_ID>

Needs: pip install psutil
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from salt import get_student_params

try:
    import psutil
except ImportError:
    psutil = None


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def linux_cache_sizes():
    sizes = {}
    base = "/sys/devices/system/cpu/cpu0/cache"
    if os.path.isdir(base):
        for idx in sorted(os.listdir(base)):
            d = os.path.join(base, idx)
            lvl_f, typ_f, sz_f, line_f = (
                os.path.join(d, "level"),
                os.path.join(d, "type"),
                os.path.join(d, "size"),
                os.path.join(d, "coherency_line_size"),
            )
            if not os.path.isfile(lvl_f):
                continue
            level = open(lvl_f).read().strip()
            typ = open(typ_f).read().strip() if os.path.isfile(typ_f) else ""
            size = open(sz_f).read().strip() if os.path.isfile(sz_f) else ""
            line = open(line_f).read().strip() if os.path.isfile(line_f) else ""
            key = f"L{level}" + ("d" if typ == "Data" else "i" if typ == "Instruction" else "")
            sizes[key] = size
            sizes.setdefault("cache_line_size", line)
    return sizes


def get_cpu_info():
    info = {
        "model_name": platform.processor() or "unknown",
        "microarchitecture": "unknown",
        "base_freq_mhz": None,
        "physical_cores": None,
        "logical_threads": os.cpu_count(),
        "smt_enabled": None,
        "l1d_cache": None,
        "l1i_cache": None,
        "l2_cache": None,
        "l3_cache": None,
        "cache_line_size": 64,
    }

    system = platform.system()

    if system == "Linux":
        cpuinfo = _sh("cat /proc/cpuinfo")
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        if m:
            info["model_name"] = m.group(1).strip()
        m = re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
        if m:
            info["base_freq_mhz"] = float(m.group(1))
        caches = linux_cache_sizes()
        info["l1d_cache"] = caches.get("L1d")
        info["l1i_cache"] = caches.get("L1i")
        info["l2_cache"] = caches.get("L2")
        info["l3_cache"] = caches.get("L3")
        if caches.get("cache_line_size"):
            info["cache_line_size"] = int(caches["cache_line_size"])
        lscpu = _sh("lscpu")
        m = re.search(r"^Thread\(s\) per core:\s*(\d+)", lscpu, re.M)
        smt = int(m.group(1)) if m else None
        m = re.search(r"^Core\(s\) per socket:\s*(\d+)", lscpu, re.M)
        cores_per_socket = int(m.group(1)) if m else None
        m = re.search(r"^Socket\(s\):\s*(\d+)", lscpu, re.M)
        sockets = int(m.group(1)) if m else 1
        if cores_per_socket and sockets:
            info["physical_cores"] = cores_per_socket * sockets
        info["smt_enabled"] = (smt is not None and smt > 1)

    elif system == "Darwin":
        out = _sh("sysctl -a | grep -E 'machdep.cpu|hw.ncpu|hw.l1|hw.l2|hw.l3|hw.cachelinesize|hw.physicalcpu|hw.logicalcpu'")
        info["raw_sysctl"] = out
        m = re.search(r"machdep\.cpu\.brand_string:\s*(.+)", out)
        if m:
            info["model_name"] = m.group(1).strip()
        m = re.search(r"hw\.physicalcpu:\s*(\d+)", out)
        if m:
            info["physical_cores"] = int(m.group(1))
        m = re.search(r"hw\.logicalcpu:\s*(\d+)", out)
        if m:
            info["logical_threads"] = int(m.group(1))
        m = re.search(r"hw\.l1dcachesize:\s*(\d+)", out)
        if m:
            info["l1d_cache"] = m.group(1)
        m = re.search(r"hw\.l2cachesize:\s*(\d+)", out)
        if m:
            info["l2_cache"] = m.group(1)
        m = re.search(r"hw\.l3cachesize:\s*(\d+)", out)
        if m:
            info["l3_cache"] = m.group(1)
        m = re.search(r"hw\.cachelinesize:\s*(\d+)", out)
        if m:
            info["cache_line_size"] = int(m.group(1))
        info["smt_enabled"] = (info["logical_threads"] or 1) > (info["physical_cores"] or 1)

    elif system == "Windows":
        ps = _sh(
            'powershell -Command "Get-CimInstance Win32_Processor | '
            'Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | Format-List"'
        )
        info["raw_powershell"] = ps
        m = re.search(r"Name\s*:\s*(.+)", ps)
        if m:
            info["model_name"] = m.group(1).strip()
        m = re.search(r"NumberOfCores\s*:\s*(\d+)", ps)
        if m:
            info["physical_cores"] = int(m.group(1))
        m = re.search(r"NumberOfLogicalProcessors\s*:\s*(\d+)", ps)
        if m:
            info["logical_threads"] = int(m.group(1))
        m = re.search(r"MaxClockSpeed\s*:\s*(\d+)", ps)
        if m:
            info["base_freq_mhz"] = float(m.group(1))
        info["smt_enabled"] = (info["logical_threads"] or 1) > (info["physical_cores"] or 1)

    # psutil fallback/cross-check
    if psutil is not None:
        info["physical_cores"] = info["physical_cores"] or psutil.cpu_count(logical=False)
        info["logical_threads"] = info["logical_threads"] or psutil.cpu_count(logical=True)
        freqs = psutil.cpu_freq()
        if freqs and not info["base_freq_mhz"]:
            info["base_freq_mhz"] = freqs.min or freqs.current

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    cpu = get_cpu_info()
    host_profile = {
        "cpu": cpu,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "build": platform.python_build(),
        },
    }
    with open("results/host_profile.json", "w") as f:
        json.dump(host_profile, f, indent=2)

    params = get_student_params(args.student_id)
    with open("results/salted_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print("Wrote results/host_profile.json")
    print(json.dumps(host_profile, indent=2))
    print("\nWrote results/salted_params.json")
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
