"""
Solar-model files: the B16, BS05 and BP00 models read exactly as in v1.6, and the B23 / SF-III layout
(struct+nu_SF3_<composition>.dat with fluxes_SF3_<composition>.dat): layout detection, column
mapping, normalisation of the production fractions, electron-density units, radius grid, fluxes,
and the survival-probability shifts between models.

The GS98 and AGSS09 compositions are shipped in Data/B23_SF3/. If the environment variable
PEANUTS_SF3_DIR points to the SolarModels/ directory of the Zenodo archive (10.5281/zenodo.10822316),
the six compositions are tested; otherwise the tests of the other four are skipped.
"""

import json
import os
import sys

import numpy as np

from _common import ROOT, DATA, peanuts
from peanuts.solar import SolarModel, Psolar, solar_flux_mass, detect_layout, sf3_flux_file, SF3_SOURCES

sys.path.insert(0, os.path.join(ROOT, "tests", "exactness"))
from solar_model_reference import fingerprint, MODELS  # noqa: E402

SF3_DATA = os.path.join(DATA, "B23_SF3")
SF3_SHIPPED = ["GS98", "AGSS09"]
SF3_ALL = ["GS98", "AGSS09", "C11", "AAG21", "MB22m", "MB22p"]

# Column numbers of the SF3 layout as documented in the archive ReadMe and its header line
SF3_COLUMNS = {"R_sun": 0, "nu_pp": 9, "nu_pep": 10, "nu_hep": 11, "nu_7Be": 12, "nu_8B": 13,
               "nu_13N": 14, "nu_15O": 15, "nu_17F": 16, "log_ne": 17}

# Reference point of the solar-model survey: dm21 = 7.5e-5 eV^2, sin^2 th12 = 0.307, sin^2 th13 = 0.022
TH12, TH13, DM21, DM3L = np.arcsin(np.sqrt(0.307)), np.arcsin(np.sqrt(0.022)), 7.5e-5, 2.5e-3

# Shift of the 8B P_ee at 10 MeV relative to B23-GS98, estimated independently from the raw tables
# (constant-density theta12 in matter at the production point, production-averaged)
SURVEY_DPEE_8B = {"bs2005opflux.dat": -1.26e-3, "nudistr_b16_gs98.dat": -8.9e-5, "nudistr_b16_agss09.dat": -2.8e-4,
                  "AGSS09": -1.6e-4, "MB22p": 4.1e-5, "MB22m": 3.1e-5, "AAG21": -1.8e-4, "C11": -1.5e-4}
# Central electron density and 8B-weighted electron density of the survey, in N_A cm^-3
SURVEY_NE = {"GS98": (100.82, 87.79), "AGSS09": (100.72, 87.97), "MB22p": (100.76, 87.74),
             "MB22m": (100.79, 87.75), "AAG21": (100.82, 88.00), "C11": (100.87, 87.97)}


def sf3_files(comps):
    """Paths of the SF3 profile files available for the given compositions."""
    out = {}
    extra = os.environ.get("PEANUTS_SF3_DIR")
    for c in comps:
        for d in [SF3_DATA] + ([extra] if extra else []):
            p = os.path.join(d, f"struct+nu_SF3_{c}.dat")
            if os.path.isfile(p):
                out[c] = p
                break
    return out


def raw_table(path):
    """Header names and rows of an SF3 profile file, parsed with numpy."""
    with open(path) as fh:
        names = fh.readline().split()[1:]
    return names, np.loadtxt(path, skiprows=1)


def pee_8B(sm, E=10.0):
    pmns = peanuts.PMNS(TH12, TH13, 0.785398, 0.0)
    return Psolar(pmns, DM21, DM3L, E, sm.radius(), sm.density(), sm.fraction("8B"))[0]


def expect_exit(f, *a, **kw):
    try:
        f(*a, **kw)
    except SystemExit:
        return
    raise AssertionError(f"{f.__name__}{a} did not exit")


def test_legacy_models_identical_to_v1_6():
    """Radius, density, fractions, fluxes and weights of the shipped B16/BS05/BP00 models, bit for bit."""
    with open(os.path.join(ROOT, "tests", "data", "solar_models_v1_6.json")) as fh:
        ref = json.load(fh)
    now = fingerprint(ROOT, MODELS)
    assert sorted(now) == sorted(ref)
    for name in ref:
        assert now[name] == ref[name], name


def test_explicit_layout_equals_detected_layout():
    for name in MODELS + [os.path.join("B23_SF3", f"struct+nu_SF3_{c}.dat") for c in SF3_SHIPPED]:
        path = os.path.join(DATA, name)
        auto = SolarModel(solar_model_file=path)
        explicit = SolarModel(solar_model_file=path, layout=auto.layout)
        assert auto.layout == detect_layout(path) and auto.layout is not None
        assert auto.flux_file == explicit.flux_file
        assert np.array_equal(auto.radius(), explicit.radius()) and np.array_equal(auto.density(), explicit.density())
        for fr in SF3_SOURCES:
            assert np.array_equal(auto.fraction(fr), explicit.fraction(fr)) and auto.flux(fr) == explicit.flux(fr)


def test_layout_detection():
    assert detect_layout("/x/struct+nu_SF3_GS98.dat") == "sf3"
    assert detect_layout("/x/b16/struct+nu_SF3_AAG21.dat") == "sf3"  # the SF3 name wins over a "b16" directory
    assert detect_layout("/x/nudistr_b16_gs98.dat") == "b16"
    assert detect_layout("/x/bs2005opflux.dat") == "bs2005"
    assert detect_layout("/x/bp00flux.dat") == "bp00"
    assert detect_layout("/x/model.dat") is None
    assert sf3_flux_file(os.path.join(SF3_DATA, "struct+nu_SF3_GS98.dat")) == os.path.join(SF3_DATA, "fluxes_SF3_GS98.dat")
    assert sf3_flux_file(os.path.join(SF3_DATA, "struct+nu_SF3_NONE.dat")) is None
    expect_exit(SolarModel, solar_model_file=os.path.join(SF3_DATA, "struct+nu_SF3_GS98.dat"), layout="sf4")
    # an SF3 profile with a non-SF3 flux file, and an unknown file without columns
    expect_exit(SolarModel, solar_model_file=os.path.join(SF3_DATA, "struct+nu_SF3_GS98.dat"),
                flux_file=os.path.join(DATA, "fluxes_b16.dat"))
    expect_exit(SolarModel, solar_model_file=os.path.join(DATA, "SNO_8B.csv"))


def test_sf3_column_mapping():
    """Every column read by SolarModel is the column of the same name in the file."""
    for comp, path in sf3_files(SF3_ALL).items():
        names, a = raw_table(path)
        assert {n: names.index(n) for n in SF3_COLUMNS} == SF3_COLUMNS, comp
        sm = SolarModel(solar_model_file=path)
        assert sm.layout == "sf3"
        assert np.array_equal(sm.radius(), a[:, 0]), comp
        # 10**log_ne with log_ne parsed by pandas; numpy's parser differs by at most one unit in the last place
        assert np.allclose(sm.density(), 10 ** a[:, 17], rtol=5e-16, atol=0), comp
        for s in SF3_SOURCES:
            x, y = sm.fraction(s), a[:, SF3_COLUMNS["nu_" + s]]
            assert np.all(np.abs(x - y) <= 2.3e-16 * np.abs(y)), (comp, s)
        # the mapping is not that of the file order: the density column comes after the fractions
        assert not np.allclose(sm.density(), 10 ** a[:, 9])


def test_sf3_radius_grid_and_normalisation():
    for comp, path in sf3_files(SF3_ALL).items():
        sm = SolarModel(solar_model_file=path)
        r = sm.radius()
        assert len(r) == 2001 and r[0] == 0.0 and r[-1] == 1.0, comp
        assert np.abs(np.diff(r) - 5e-4).max() < 1e-12, comp
        for s in SF3_SOURCES:
            f = sm.fraction(s)
            # production fractions per unit R_sun, normalised to unit integral; the negative entries are
            # interpolation noise of the published table (at r = 0 and in the outer tail)
            assert abs(np.trapz(f, r) - 1) < 1e-6, (comp, s, np.trapz(f, r))
            assert f.min() > -3e-11 and f.max() > 1, (comp, s)
            assert np.trapz(np.clip(f, None, 0), r) > -1e-13, (comp, s)


def test_sf3_density_units():
    """n_e in mol/cm^3 (N_A cm^-3), as for B16: central value and 8B-weighted mean."""
    b16 = SolarModel(solar_model_file=os.path.join(DATA, "nudistr_b16_gs98.dat"))
    for comp, path in sf3_files(SF3_ALL).items():
        sm = SolarModel(solar_model_file=path)
        r, n, f = sm.radius(), sm.density(), sm.fraction("8B")
        n0, n8 = SURVEY_NE[comp]
        assert abs(n[0] - n0) < 0.006, (comp, n[0])
        assert abs(np.trapz(n * f, r) / np.trapz(f, r) - n8) < 0.006, comp
        assert abs(n[0] / b16.density()[0] - 1) < 5e-3, comp
        assert n[-1] < 1e-9 and np.all(np.diff(n[r < 0.9]) < 0), comp


def test_sf3_fluxes():
    for comp, path in sf3_files(SF3_ALL).items():
        fluxpath = os.path.join(os.path.dirname(path), f"fluxes_SF3_{comp}.dat")
        if not os.path.isfile(fluxpath):
            continue
        sm = SolarModel(solar_model_file=path)
        assert sm.flux_file == fluxpath
        with open(fluxpath) as fh:
            lines = fh.readlines()
        names, values = lines[1].split(), lines[2].split()
        assert values[-1] == "Flux"
        for s in SF3_SOURCES:
            assert sm.flux(s) == float(values[names.index(s)]), (comp, s)
        assert 5.9e10 < sm.flux("pp") < 6.1e10 and 4e6 < sm.flux("8B") < 6e6


def test_sf3_weights_are_normalised():
    rng = np.random.default_rng(41)
    for comp, path in sf3_files(SF3_SHIPPED).items():
        sm = SolarModel(solar_model_file=path)
        r, n = sm.radius(), sm.density()
        for _ in range(30):
            th12, th13 = rng.uniform(0.45, 0.75), rng.uniform(0.0, 0.25)
            dm21, dm3l = 10 ** rng.uniform(-5.5, -3.5), rng.uniform(2.2e-3, 2.7e-3) * rng.choice([-1.0, 1.0])
            for s in SF3_SOURCES:
                w = solar_flux_mass(th12, th13, dm21, dm3l, 10 ** rng.uniform(-1.5, 1.3), r, n, sm.fraction(s))
                assert np.all(w >= -1e-15) and abs(w.sum() - 1) < 1e-13, (comp, s, w)


def test_pee_shifts_against_survey():
    """8B P_ee at 10 MeV: shifts of BS05, B16 and the B23 compositions relative to B23-GS98, against the
    independent estimate of the survey (5 % or 3e-6)."""
    files = sf3_files(SF3_ALL)
    ref = pee_8B(SolarModel(solar_model_file=files["GS98"]))
    models = {k: os.path.join(DATA, k) for k in ["bs2005opflux.dat", "nudistr_b16_gs98.dat", "nudistr_b16_agss09.dat"]}
    models.update({c: p for c, p in files.items() if c != "GS98"})
    for key, path in models.items():
        d = pee_8B(SolarModel(solar_model_file=path)) - ref
        assert abs(d - SURVEY_DPEE_8B[key]) <= 0.05 * abs(SURVEY_DPEE_8B[key]) + 3e-6, (key, d)
