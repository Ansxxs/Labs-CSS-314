"""
salt.py -- Milestone 0, Task 0.2
Deterministically derives this student's problem parameters from their
Student ID, per the practicum spec (SHA-256 of the ID string).
"""
import hashlib


def get_student_params(student_id: str):
    h = int(hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8], 16)
    # Salted matrix dimension N between 1024 and 1536 (multiple of 64)
    N = 1024 + ((h % 9) * 64)
    # Salted stride benchmark buffer size: 2^24 elements (~134 MB double-precision)
    buffer_elements = 16_777_216
    # Seed for deterministic random number generation
    seed = h % 100_007
    return {
        "student_id": student_id,
        "N": N,
        "buffer_elements": buffer_elements,
        "seed": seed,
    }


if __name__ == "__main__":
    import sys, json
    sid = sys.argv[1] if len(sys.argv) > 1 else "00000000"
    print(json.dumps(get_student_params(sid), indent=2))
