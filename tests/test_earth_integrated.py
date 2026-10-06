"""
Pearth_integrated: the compiled path with its caches against the direct sum over the nadir samples
(Pearth_integrated_reference), for every density model, exposure type, depth and neutrino type;
independence of the result from the call history; invalidation of the caches; linearity in the
neutrino state; probability conservation.
"""

import os
import tempfile

import numpy as np

from _common import peanuts, POINTS, SNO_EXPOSURE, random_point
from peanuts import earth as pe


def both(state, density, point, E, depth, **kw):
    th12, th13, th23, d, dm21, dm3l = point
    fast = peanuts.Pearth_integrated(state, density, peanuts.PMNS(th12, th13, th23, d), dm21, dm3l, E, depth, **kw)
    ref = pe.Pearth_integrated_reference(state, density, peanuts.PMNS(th12, th13, th23, d), dm21, dm3l, E, depth, **kw)
    return fast, ref


def tabulated_density_file():
    """A tabulated profile (constant-density layers) derived from the default PREM parametrisation."""
    ed = peanuts.EarthDensity()
    r = np.linspace(0.02, 1.0, 50)
    n = np.array([ed.call(x, 0.0) for x in r - 0.01])
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    f.write("# rj, alpha\n")
    for x, v in zip(r, n):
        f.write(f"{x:.6f},{v:.8f}\n")
    f.close()
    return f.name


def test_compiled_equals_direct_sum():
    tab = tabulated_density_file()
    densities = {"default": peanuts.EarthDensity(), "custom": peanuts.EarthDensity(custom_density=True),
                 "tabulated": peanuts.EarthDensity(density_file=tab, tabulated_density=True)}
    rng = np.random.default_rng(21)
    states = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]), np.array([0.35, 0.55, 0.10])]
    n = 0
    for name, ed in densities.items():
        for k, point in enumerate(POINTS[:3] + [random_point(rng)]):
            for depth in [0.0, 1400.0, 2092.0]:
                for daynight in ["day", "night", ""]:
                    for anti in [False, True]:
                        E = 10 ** rng.uniform(-1, 1.3)
                        s = states[n % len(states)]
                        fast, ref = both(s, ed, point, E, depth, antinu=anti, ns=480, from_file=SNO_EXPOSURE,
                                         angle="CosZenith", daynight=daynight)
                        assert np.array_equal(fast, ref), (name, k, depth, daynight, anti, fast, ref)
                        n += 1
    os.unlink(tab)


def test_latitude_exposure():
    ed = peanuts.EarthDensity()
    for daynight in [None, "day", "night"]:
        fast, ref = both(np.array([0.6, 0.3, 0.1]), ed, POINTS[1], 8.0, 1400.0, lam=0.7405, ns=40, daynight=daynight)
        assert np.array_equal(fast, ref)


def test_history_independence():
    """A random sequence of calls that revisits parameters, energies, depths, exposures and states
    in arbitrary order: each result must equal a fresh direct sum."""
    rng = np.random.default_rng(22)
    ed = peanuts.EarthDensity()
    ed2 = peanuts.EarthDensity(custom_density=True)
    pts = POINTS[:3]
    energies = [0.7, 5.0, 12.0]
    expected = {}
    for _ in range(150):
        point = pts[rng.integers(3)]
        E = energies[rng.integers(3)]
        depth = [1400.0, 2092.0][rng.integers(2)]
        daynight = ["day", "night", ""][rng.integers(3)]
        density = [ed, ed2][rng.integers(2)]
        state = np.zeros(3); state[rng.integers(3)] = 1.0
        if rng.integers(4) == 0:
            state = rng.dirichlet([1, 1, 1])
        key = (point, E, depth, daynight, id(density), state.tobytes())
        fast = peanuts.Pearth_integrated(state, density, peanuts.PMNS(*point[:4]), point[4], point[5], E, depth,
                                         ns=480, from_file=SNO_EXPOSURE, angle="CosZenith", daynight=daynight)
        if key not in expected:
            expected[key] = pe.Pearth_integrated_reference(state, density, peanuts.PMNS(*point[:4]), point[4], point[5], E, depth,
                                                           ns=480, from_file=SNO_EXPOSURE, angle="CosZenith", daynight=daynight)
        assert np.array_equal(fast, expected[key]), key


def test_exposure_file_change_is_detected():
    raw = open(SNO_EXPOSURE).read().splitlines()
    header, values = raw[:9], [float(v) for v in raw[9:]]
    f = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False)
    f.write("\n".join(header + [repr(float(v)) for v in values]) + "\n"); f.close()
    ed = peanuts.EarthDensity()
    s = np.array([0.3, 0.6, 0.1])
    kw = dict(ns=480, from_file=f.name, angle="CosZenith", daynight="night")
    first, ref = both(s, ed, POINTS[0], 9.0, 1730.0, **kw)
    assert np.array_equal(first, ref)
    # Rewrite the file with a different exposure and a later modification time
    with open(f.name, "w") as g:
        g.write("\n".join(header + [repr(float(v * (1 + 0.5 * np.sin(i)))) for i, v in enumerate(values)]) + "\n")
    st = os.stat(f.name)
    os.utime(f.name, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    second, ref2 = both(s, ed, POINTS[0], 9.0, 1730.0, **kw)
    assert np.array_equal(second, ref2) and not np.array_equal(first, second)
    os.unlink(f.name)


def test_linearity_in_the_state():
    """P(w) = sum_j w_j P(e_j), the property behind the reuse of the projectors."""
    ed = peanuts.EarthDensity()
    rng = np.random.default_rng(23)
    for point in POINTS:
        for E in [0.4, 6.0, 15.0]:
            w = rng.dirichlet([1, 1, 1])
            kw = dict(ns=480, from_file=SNO_EXPOSURE, angle="CosZenith", daynight="")
            pw = peanuts.Pearth_integrated(w, ed, peanuts.PMNS(*point[:4]), point[4], point[5], E, 1730.0, **kw)
            basis = [peanuts.Pearth_integrated(np.eye(3)[j], ed, peanuts.PMNS(*point[:4]), point[4], point[5], E, 1730.0, **kw) for j in range(3)]
            comb = sum(w[j] * basis[j] for j in range(3))
            assert np.abs(pw - comb).max() <= 1e-14 * np.abs(pw).max()


def test_probability_conservation():
    """Day samples: the evolutor is exactly unitary, so the flavour probabilities of a normalised
    state sum to the exposure integral (to 1e-13). Night samples: the first-order evolutor is
    unitary up to second order in the density perturbation; the defect, the same as in v1.5, grows
    at small dm21 and high energy (2e-3 at dm21 = 1e-5 eV^2, 15 MeV) and must stay below 5e-3."""
    ed = peanuts.EarthDensity()
    ex = {dn: pe.cached_exposure(ns=480, from_file=SNO_EXPOSURE, angle="CosZenith", daynight=dn) for dn in ["day", "night"]}
    for point in POINTS:
        for E in [0.5, 5.0, 15.0]:
            for dn, tol in [("day", 1e-13), ("night", 5e-3)]:
                norm = np.sum(ex[dn].weights) * np.pi / 480
                p = peanuts.Pearth_integrated(np.array([0.2, 0.5, 0.3]), ed, peanuts.PMNS(*point[:4]), point[4], point[5], E, 1730.0,
                                              ns=480, from_file=SNO_EXPOSURE, angle="CosZenith", daynight=dn)
                assert abs(p.sum() / norm - 1) < tol, (point, E, dn, p.sum() / norm - 1)


def test_fallbacks_and_errors():
    ed = peanuts.EarthDensity()
    point = POINTS[0]
    pmns = peanuts.PMNS(*point[:4])
    kw = dict(ns=480, from_file=SNO_EXPOSURE, angle="CosZenith", daynight="day")
    # a list state takes the direct path, as in v1.5
    try:
        r = peanuts.Pearth_integrated([1., 0., 0.], ed, pmns, point[4], point[5], 8.0, 1730.0, **kw)
    except Exception as e:  # v1.5 behaviour: numba cannot type a reflected list here
        r = type(e).__name__
    try:
        r_ref = pe.Pearth_integrated_reference([1., 0., 0.], ed, pmns, point[4], point[5], 8.0, 1730.0, **kw)
    except Exception as e:
        r_ref = type(e).__name__
    assert type(r) == type(r_ref) and (isinstance(r, str) and r == r_ref or np.array_equal(r, r_ref))
    # wrong state size: v1.5 exits
    try:
        peanuts.Pearth_integrated(np.ones(4), ed, pmns, point[4], point[5], 8.0, 1730.0, **kw)
        raise AssertionError("no exit for a state of size 4")
    except SystemExit:
        pass
    # ns different from the number of samples in the file: v1.5 exits
    try:
        peanuts.Pearth_integrated(np.ones(3), ed, pmns, point[4], point[5], 8.0, 1730.0, ns=400, from_file=SNO_EXPOSURE, angle="CosZenith")
        raise AssertionError("no exit for ns != samples in file")
    except SystemExit:
        pass


def test_no_recompilation_across_points():
    """The compiled kernels are specialised once; new parameter points, energies, orderings and
    neutrino types must not add signatures (a new signature would mean a JIT compilation per call)."""
    ed = peanuts.EarthDensity()
    kw = dict(ns=480, from_file=SNO_EXPOSURE, angle="CosZenith", daynight="")
    def run(point, E, anti):
        peanuts.Pearth_integrated(np.array([0.2, 0.5, 0.3]), ed, peanuts.PMNS(*point[:4]), point[4], point[5], E, 1730.0, antinu=anti, **kw)
    run(POINTS[0], 5.0, False); run(POINTS[0], 5.0, True)
    before = (len(pe.mass_projectors.signatures), len(pe.integrate_projectors.signatures), len(pe.path_geometry.signatures))
    for point in POINTS:
        for E in [0.3, 7.0, 19.0]:
            for anti in [False, True]:
                run(point, E, anti)
    after = (len(pe.mass_projectors.signatures), len(pe.integrate_projectors.signatures), len(pe.path_geometry.signatures))
    assert before == after, (before, after)
