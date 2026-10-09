"""
Command-line programs: run_peanuts.py on every example and run_prob_sun.py / run_prob_earth.py.
If the environment variable PEANUTS_REFERENCE points to another PEANUTS root (e.g. v1.5), the
same commands are run there and the printed numbers must agree to 1e-12 relative, or both runs
must fail with the same error.
"""

import os
import re
import subprocess
import sys

from _common import ROOT

COMMANDS = [["run_peanuts.py", "-f", f"examples/{f}"] for f in sorted(os.listdir(os.path.join(ROOT, "examples"))) if f.endswith(".yaml")] + [
    ["run_prob_sun.py", "10", "8B", "0.5903", "0.1503", "0.8587", "3.4034", "7.42e-5", "2.51e-3"],
    ["run_prob_sun.py", "0.5", "pp", "0.5903", "0.1503", "0.8587", "3.4034", "7.42e-5", "-2.49e-3"],
    ["run_prob_earth.py", "-m", "0.5,0.3,0.2", "10", "0.3", "1400", "0.5903", "0.1503", "0.8587", "3.4034", "7.42e-5", "2.51e-3"],
    ["run_prob_earth.py", "-f", "1,0,0", "--antinu", "5", "2.5", "1400", "0.5903", "0.1503", "0.8587", "3.4034", "7.42e-5", "-2.49e-3"],
    ["run_prob_earth.py", "-m", "0.2,0.2,0.6", "15", "0.05", "0", "0.6", "0.0", "0.8", "0.0", "5e-5", "2.5e-3"],
]

NUMBER = re.compile(r"[-+]?\d+\.\d*(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+")


def run(root, cmd):
    p = subprocess.run([sys.executable] + cmd, cwd=root, capture_output=True, text=True, timeout=1800)
    err = [l for l in p.stderr.splitlines() if "Error" in l or "Exception" in l]
    return p.returncode, p.stdout, (err[-1] if err else "")


def numbers(text):
    # drop the banner, which prints the git tag of the working directory
    body = text.split("Running PEANUTS...")[-1] if "Running PEANUTS..." in text else text
    return [float(x) for x in NUMBER.findall(body)]


def test_cli():
    ref_root = os.environ.get("PEANUTS_REFERENCE")
    for cmd in COMMANDS:
        code, out, err = run(ROOT, cmd)
        if ref_root:
            rcode, rout, rerr = run(ref_root, cmd)
            assert (code == 0) == (rcode == 0), (cmd, err, rerr)
            if code == 0:
                a, b = numbers(out), numbers(rout)
                assert len(a) == len(b) and len(a) > 0, (cmd, len(a), len(b))
                for x, y in zip(a, b):
                    assert abs(x - y) <= 1e-12 * max(abs(x), abs(y), 1e-300), (cmd, x, y)
            else:
                assert err.split(":")[0] == rerr.split(":")[0], (cmd, err, rerr)
        elif cmd[-1] not in ("examples/SNO_test.yaml", "examples/exposure_test.yaml"):
            # the two examples with exposure_normalized: true fail with SciPy >= 1.14 in v1.5 as well
            assert code == 0 and numbers(out), (cmd, err)
