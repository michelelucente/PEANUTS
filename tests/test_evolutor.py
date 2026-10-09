"""
The evolutor of a density shell against independent calculations: the matrix exponential for
constant density, a high-precision ODE integration for the polynomial density profiles of the
Earth model, numerical quadrature for Iab, numpy eigenvalues for lambdas, the PDG vacuum formula.
"""

import warnings

import numpy as np
import scipy.linalg as sl
from scipy.integrate import quad, solve_ivp

from _common import peanuts, POINTS, R_E, HBARC, pdg_pmns, squared_masses, random_point
from peanuts import evolutor as ev
from peanuts import integration as ig
from peanuts.potentials import MatterPotential


def hamiltonian(pmns, dm21, dm3l, E, n, antinu):
    ki = ev.kinetic_terms(dm21, dm3l, E)
    U = pmns.U.conjugate() if antinu else pmns.U
    Hk = ev.kinetic_hamiltonian(U, ki)
    return ki, Hk, Hk + np.diag([MatterPotential(n, antinu), 0, 0])


def test_constant_density_against_matrix_exponential():
    """For constant density the evolutor of Upert is exact: it must equal expm(-i H L). The
    rounding error grows with the phase |lambda| L, so the tolerance is 2e-13 per radian."""
    rng = np.random.default_rng(11)
    for _ in range(400):
        th12, th13, th23, d, dm21, dm3l = random_point(rng)
        pmns = peanuts.PMNS(th12, th13, th23, d)
        E, n, L, anti = 10 ** rng.uniform(-1.5, 2), rng.uniform(0, 7), rng.uniform(0.01, 2), bool(rng.integers(2))
        ki, Hk, H = hamiltonian(pmns, dm21, dm3l, E, n, anti)
        u = ev.Upert_kinetic(ki, Hk, th12, th13, L, 0., n, 0., 0., anti)
        u_ref = peanuts.evolutor.Upert(dm21, dm3l, pmns, E, L, 0., n, 0., 0., anti)
        phase = max(1.0, np.abs(np.linalg.eigvals(H)).max() * L)
        assert np.abs(u - sl.expm(-1j * H * L)).max() <= 2e-13 * phase
        assert np.abs(u @ u.conj().T - np.eye(3)).max() <= 2e-13 * phase
        assert np.array_equal(u, u_ref)


def test_lambdas_are_the_eigenvalues_of_T():
    rng = np.random.default_rng(12)
    for _ in range(300):
        th12, th13, th23, d, dm21, dm3l = random_point(rng)
        pmns = peanuts.PMNS(th12, th13, th23, d)
        E, n, anti = 10 ** rng.uniform(-1.5, 2), rng.uniform(0, 13), bool(rng.integers(2))
        ki, Hk, H = hamiltonian(pmns, dm21, dm3l, E, n, anti)
        T = H - np.trace(H) / 3 * np.eye(3)
        lam = ig.lambdas(ig.c0(ki, th12, th13, n, anti), ig.c1(ki, th12, th13, n, anti))
        ref = np.sort(np.linalg.eigvalsh(T))
        assert np.abs(np.sort(lam.real) - ref).max() <= 1e-10 * np.abs(ref).max()
        assert np.abs(lam.imag).max() <= 1e-10 * np.abs(ref).max()


def test_Iab_against_quadrature():
    """Iab(la, lb) = int_x1^x2 exp(-i la (x2-x)) (atilde + b x^2 + c x^4) exp(-i lb (x-x1)) dx.
    Analytic branch, with |Dl (x2-x1)| >= 0.1: agreement to 1e-10 of the integrand scale.
    Taylor branch (|Dl/(la+lb)| < 1e-2): second-order truncation, agreement to 1e-6."""
    rng = np.random.default_rng(13)
    warnings.simplefilter("ignore")
    for branch in ["analytic", "taylor"]:
        n = 0
        while n < 100:
            la = complex(rng.uniform(5, 50) * rng.choice([-1, 1]), 0)
            lb = la * (1 + 1e-3 * rng.uniform(-1, 1)) if branch == "taylor" else complex(rng.uniform(-50, 50), 0)
            x1 = rng.uniform(0, .5); x2 = x1 + rng.uniform(.05, .5)
            b, c = rng.uniform(-20, 20), rng.uniform(-20, 20)
            at = -(b * (x2**3 - x1**3) / 3 + c * (x2**5 - x1**5) / 5) / (x2 - x1)
            if branch == "analytic" and (abs((la - lb) / (la + lb)) < 1e-2 or abs((la - lb) * (x2 - x1)) < 0.1):
                continue
            g = lambda x: np.exp(-1j * la * (x2 - x)) * (at + b * x * x + c * x**4) * np.exp(-1j * lb * (x - x1))
            q = (quad(lambda x: g(x).real, x1, x2, epsabs=1e-15, epsrel=1e-13, limit=400)[0]
                 + 1j * quad(lambda x: g(x).imag, x1, x2, epsabs=1e-15, epsrel=1e-13, limit=400)[0])
            scale = abs(b) * (x2 - x1)**3 + abs(c) * (x2 - x1)**5 + abs(at) * (x2 - x1)
            tol = 1e-10 if branch == "analytic" else 1e-6
            assert abs(ig.Iab(la, lb, at, b, c, x2, x1) - q) <= tol * scale, (branch, la, lb)
            n += 1
    assert ig.Iab(3.0 + 0j, 3.0 + 0j, 1., 2., 3., .5, .1) == 0


def test_first_order_shell_evolutor_against_ode():
    """Earth shells with a polynomial profile: the first-order evolutor must agree with a
    high-precision integration of the Schroedinger equation to 2e-3 and improve on the
    zeroth-order (average density) evolutor by at least a factor 3 (measured: 7e-4 and 4.8)."""
    ed = peanuts.EarthDensity()
    for th12, th13, th23, d, dm21, dm3l in POINTS[:2]:
        pmns = peanuts.PMNS(th12, th13, th23, d)
        for eta in [0.0, 0.3, 0.6, 1.0, 1.4]:
            params, xs = ed.parameters(eta), ed.shells_x(eta)
            for j in range(len(xs)):
                x2, x1 = xs[j], (xs[j - 1] if j > 0 else 0.)
                a, b, c = params[j, 0], params[j, 1], params[j, 2]
                if b == 0 and c == 0:
                    continue
                for E in [1., 10., 20.]:
                    ki, Hk, _ = hamiltonian(pmns, dm21, dm3l, E, 0., False)
                    f = lambda x, y: (-1j * (Hk + np.diag([MatterPotential(a + b * x * x + c * x**4, False), 0, 0])) @ y.reshape(3, 3)).ravel()
                    ode = solve_ivp(f, (x1, x2), np.eye(3, dtype=complex).ravel(), rtol=1e-11, atol=1e-12, method="DOP853").y[:, -1].reshape(3, 3)
                    u = ev.Upert_kinetic(ki, Hk, th12, th13, x2, x1, a, b, c, False)
                    nav = ev.average_density(x2, x1, a, b, c)
                    lam, M, tr = ev.spectral_decomposition(ki, Hk, th12, th13, nav, False)
                    u0 = ev.evolutor_from_spectrum(lam, M, tr, x2, x1, a - nav, 0, 0, False)
                    err1, err0 = np.abs(u - ode).max(), np.abs(u0 - ode).max()
                    assert err1 <= 2e-3 and err0 >= 3 * err1, (eta, j, E, err1, err0)


def test_vacuum_against_pdg_formula():
    """peanuts.vacuum.Pvacuum in the flavour basis against |U exp(-i m^2 L / 2E) U^dagger|^2. The
    rounding error grows with the oscillation phase (up to 2e5 rad here): tolerance 1e-13 per radian."""
    from peanuts.vacuum import Pvacuum
    rng = np.random.default_rng(14)
    for k in range(60):
        th12, th13, th23, d, dm21, dm3l = POINTS[k % len(POINTS)] if k < len(POINTS) else random_point(rng)
        pmns = peanuts.PMNS(th12, th13, th23, d)
        E, L, anti = 10 ** rng.uniform(-1, 2), 10 ** rng.uniform(0, 4), bool(rng.integers(2))
        U = pdg_pmns(th12, th13, th23, d)
        if anti:
            U = U.conj()
        phi = 0.5e-12 * squared_masses(dm21, dm3l) * L * 1e3 / E / HBARC
        S = U @ np.diag(np.exp(-1j * phi)) @ U.conj().T
        tol = 1e-13 * max(1.0, np.abs(phi).max())
        assert np.abs(peanuts.PMNS(th12, th13, th23, d).pmns - pdg_pmns(th12, th13, th23, d)).max() < 1e-15
        for a in range(3):
            state = np.zeros(3, dtype=complex); state[a] = 1
            p = Pvacuum(state, pmns, dm21, dm3l, E, L, antinu=anti, massbasis=False)
            assert np.abs(p - np.abs(S[:, a])**2).max() <= tol, (k, a)


def test_surface_detector_from_above_is_vacuum_projection():
    """At depth 0 a neutrino from above the horizon does not cross matter: P = |U|^2 w."""
    ed = peanuts.EarthDensity()
    w = np.array([0.2, 0.7, 0.1])
    for th12, th13, th23, d, dm21, dm3l in POINTS:
        pmns = peanuts.PMNS(th12, th13, th23, d)
        for eta in [np.pi / 2, 2.0, np.pi]:
            p = peanuts.Pearth(w, ed, pmns, dm21, dm3l, 10., eta, 0.)
            assert np.abs(p - np.abs(pdg_pmns(th12, th13, th23, d))**2 @ w).max() < 1e-15


def test_analytical_against_numerical_mode():
    """The analytical evolutor of PEANUTS against its own numerical (ODE) mode, Earth-crossing
    and above-horizon paths: agreement to 2e-3 (measured: up to 9e-4)."""
    ed = peanuts.EarthDensity()
    th12, th13, th23, d, dm21, dm3l = POINTS[0]
    pmns = peanuts.PMNS(th12, th13, th23, d)
    for E, eta in [(10., 0.3), (5., 1.0), (15., 0.1), (8., 2.0), (12., 0.0)]:
        w = np.array([0.3, 0.5, 0.2])
        pa = peanuts.Pearth(w, ed, pmns, dm21, dm3l, E, eta, 1400.)
        pn = peanuts.Pearth(w, ed, pmns, dm21, dm3l, E, eta, 1400., mode="numerical")
        assert np.abs(pa - pn).max() < 2e-3, (E, eta, pa, pn)


def test_errors_are_preserved():
    ed = peanuts.EarthDensity()
    pmns = peanuts.PMNS(*POINTS[0][:4])
    for eta in [-0.1, 3.2]:
        try:
            peanuts.Pearth(np.array([1., 0., 0.]), ed, pmns, 7.4e-5, 2.5e-3, 10., eta, 1400.)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for eta = {eta}")
