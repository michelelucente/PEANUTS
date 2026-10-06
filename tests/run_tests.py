#!/usr/bin/env python3
"""
Runs the PEANUTS test suite without pytest (pytest tests/ works as well): every test_* function of
the tests/test_*.py modules, in order, with its outcome and run time.

Usage: run_tests.py [module_or_test_substring ...]
"""

import glob
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

selected = sys.argv[1:]
failed, passed = [], 0
for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
    module = importlib.import_module(os.path.basename(path)[:-3])
    for name in [n for n in dir(module) if n.startswith("test_")]:
        label = f"{module.__name__}.{name}"
        if selected and not any(s in label for s in selected):
            continue
        t0 = time.time()
        try:
            getattr(module, name)()
            passed += 1
            print(f"PASS {label} ({time.time() - t0:.1f} s)", flush=True)
        except BaseException as e:  # SystemExit included
            failed.append(label)
            print(f"FAIL {label} ({time.time() - t0:.1f} s): {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
print(f"\n{passed} passed, {len(failed)} failed" + (": " + ", ".join(failed) if failed else ""))
sys.exit(1 if failed else 0)
