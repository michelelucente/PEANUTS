#!/usr/bin/env python3
"""
Fingerprint of the solar models shipped in Data/ as read by a PEANUTS installation: SHA-256 of the
radius, electron-density and production-fraction arrays (float64 bytes), the total fluxes, and the
mass-state weights solar_flux_mass at fixed points and energies, all written exactly (float.hex).

Usage: solar_model_reference.py <PEANUTS root> <output.json>

tests/data/solar_models_v1_6.json was produced with the v1.6 tag
(git archive v1.6 | tar -x -C <dir>; solar_model_reference.py <dir> tests/data/solar_models_v1_6.json);
tests/test_solar_models.py compares the working tree with it.
"""

import hashlib
import json
import os
import sys

import numpy as np

MODELS = ["nudistr_b16_gs98.dat", "nudistr_b16_agss09.dat", "bs2005opflux.dat", "bs2005agsopflux.csv", "bp00flux.dat"]
FRACTIONS = ["pp", "pep", "hep", "7Be", "8B", "13N", "15O", "17F"]
# (th12, th13, dm21, dm3l): NO and IO near the best fit, a large-dm21 and a small-dm21 point
POINTS = [(0.5903, 0.1503, 7.42e-5, 2.51e-3), (0.5903, 0.1503, 7.42e-5, -2.49e-3),
          (0.6100, 0.1500, 1.5e-4, -2.4e-3), (0.4500, 0.2200, 1.0e-5, 2.3e-3)]
ENERGIES = [0.2, 0.862, 1.44, 5.0, 10.0, 15.0]


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


def fingerprint(root, models=MODELS):
    sys.path.insert(0, root)
    from peanuts.solar import SolarModel, solar_flux_mass
    out = {}
    for name in models:
        sm = SolarModel(solar_model_file=os.path.join(root, "Data", name))
        r, n = sm.radius(), sm.density()
        entry = {"rows": int(len(r)), "radius": sha(r), "density": sha(n), "fractions": {}, "fluxes": {}, "weights": {}}
        for fr in FRACTIONS:
            if not sm.has_fraction(fr):
                continue
            entry["fractions"][fr] = sha(sm.fraction(fr))
            entry["fluxes"][fr] = float(sm.flux(fr)).hex()
            entry["weights"][fr] = [[float(w).hex() for w in solar_flux_mass(t12, t13, d21, d3l, E, r, n, sm.fraction(fr))]
                                    for (t12, t13, d21, d3l) in POINTS for E in ENERGIES]
        out[name] = entry
    return out


if __name__ == "__main__":
    root, dest = os.path.abspath(sys.argv[1]), sys.argv[2]
    with open(dest, "w") as fh:
        json.dump(fingerprint(root), fh, indent=1, sort_keys=True)
        fh.write("\n")
