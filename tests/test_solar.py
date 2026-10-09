"""
Solar mass-state weights: normalisation, vacuum limit, the reuse of th13_M, Psolar.
"""

import numpy as np

from _common import peanuts, POINTS, pdg_pmns, random_point
from peanuts.matter_mixing import th12_M, th13_M, th12_M_given_th13_M
from peanuts.solar import solar_flux_mass, Psolar

SM = peanuts.SolarModel()
FRACTIONS = ["pp", "pep", "hep", "7Be", "8B", "13N", "15O", "17F"]


def test_weights_are_normalised():
    rng = np.random.default_rng(31)
    r, n = SM.radius(), SM.density()
    for k in range(40):
        th12, th13, th23, d, dm21, dm3l = POINTS[k % len(POINTS)] if k < len(POINTS) else random_point(rng)
        for fr in FRACTIONS:
            w = solar_flux_mass(th12, th13, dm21, dm3l, 10 ** rng.uniform(-1.5, 1.3), r, n, SM.fraction(fr))
            assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-13


def test_vacuum_limit():
    """At zero density the weights are |U_ei|^2."""
    r = SM.radius()
    for th12, th13, th23, d, dm21, dm3l in POINTS:
        w = solar_flux_mass(th12, th13, dm21, dm3l, 8.0, r, np.zeros_like(r), SM.fraction("8B"))
        assert np.abs(w - np.abs(pdg_pmns(th12, th13, th23, d)[0])**2).max() < 1e-14


def test_th13_M_reuse_is_identical():
    rng = np.random.default_rng(32)
    n = SM.density()
    for _ in range(50):
        th12, th13, th23, d, dm21, dm3l = random_point(rng)
        E = 10 ** rng.uniform(-1.5, 1.3)
        t13 = th13_M(th12, th13, dm21, dm3l, E, n)
        assert np.array_equal(th12_M(th12, th13, dm21, dm3l, E, n), th12_M_given_th13_M(th12, th13, t13, dm21, dm3l, E, n))


def test_Psolar_sums_to_one():
    r, n = SM.radius(), SM.density()
    for th12, th13, th23, d, dm21, dm3l in POINTS:
        for fr in FRACTIONS:
            p = Psolar(peanuts.PMNS(th12, th13, th23, d), dm21, dm3l, 5.0, r, n, SM.fraction(fr))
            assert abs(p.sum() - 1) < 1e-13
