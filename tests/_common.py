"""
Shared helpers of the PEANUTS test suite. The tests import the `peanuts` package of the repository
that contains this directory.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import peanuts  # noqa: E402

DATA = os.path.join(ROOT, "Data")
SNO_EXPOSURE = os.path.join(DATA, "SnoCosZenith.dat")

# Speed of light times hbar in MeV m, as in peanuts.potentials
HBARC = 197.3269804e-15
R_E = 6.371e6

# Parameter points (th12, th13, th23, delta, dm21, dm3l): normal and inverted ordering, theta13 = 0,
# delta = 0, and points away from the best fit
POINTS = [
    (0.5903, 0.1503, 0.8587, 3.4034, 7.42e-5, 2.51e-3),
    (0.5903, 0.1503, 0.8587, 3.4034, 7.42e-5, -2.49e-3),
    (0.5800, 0.0, 0.7854, 0.0, 5.0e-5, 2.5e-3),
    (0.6100, 0.1500, 0.9000, 4.7, 1.5e-4, -2.4e-3),
    (0.4500, 0.2200, 0.6500, 1.2, 1.0e-5, 2.3e-3),
]


def random_point(rng):
    dm3l = rng.uniform(2.2e-3, 2.7e-3) * rng.choice([-1.0, 1.0])
    return (rng.uniform(0.45, 0.75), rng.uniform(0.0, 0.25), rng.uniform(0.6, 1.0),
            rng.uniform(0, 2 * np.pi), 10 ** rng.uniform(-5.5, -3.5), dm3l)


def pdg_pmns(th12, th13, th23, delta):
    """PMNS matrix in the PDG parametrisation, written independently of peanuts.pmns."""
    s12, c12, s13, c13, s23, c23 = np.sin(th12), np.cos(th12), np.sin(th13), np.cos(th13), np.sin(th23), np.cos(th23)
    e = np.exp(1j * delta)
    return np.array([
        [c12 * c13, s12 * c13, s13 / e],
        [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13],
    ])


def squared_masses(dm21, dm3l):
    """m_i^2 up to a common constant, in eV^2 (dm3l = dm31 for NO, dm32 for IO)."""
    return np.array([0.0, dm21, dm3l]) if dm3l > 0 else np.array([-dm21, 0.0, dm3l])
