#!/usr/bin/env python3
"""
Compares two .npz files written by workload.py. Reports, for every output, whether the values are
bit-for-bit identical, equal as numbers (which identifies +0 and -0), or different, and the largest
relative difference. Exit status 0 only if every output is equal as numbers.

Usage: compare.py <reference.npz> <new.npz>
"""

import sys
import numpy as np

ref = np.load(sys.argv[1])
new = np.load(sys.argv[2])

missing = sorted(set(ref.files) ^ set(new.files))
bitwise = numeric = differ = 0
worst = (0.0, None)
for k in ref.files:
    if k not in new.files:
        continue
    a, b = ref[k], new[k]
    if a.shape != b.shape or a.dtype != b.dtype:
        differ += 1
        print(f"shape/dtype mismatch {k}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
        continue
    if a.tobytes() == b.tobytes():
        bitwise += 1
    elif np.array_equal(a, b, equal_nan=True):
        numeric += 1
    else:
        differ += 1
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))
        if rel > worst[0]:
            worst = (rel, k)
        if differ <= 10:
            print(f"DIFFERENT {k}: max relative difference {rel:.3e}")

print(f"outputs: {len(ref.files)}; bit-for-bit identical: {bitwise}; equal up to the sign of zero: {numeric}; "
      f"different: {differ}; missing on one side: {len(missing)}")
if differ:
    print(f"largest relative difference {worst[0]:.3e} in {worst[1]}")
sys.exit(0 if (differ == 0 and not missing) else 1)
