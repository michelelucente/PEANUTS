#!/usr/bin/env python3
"""
How the BLAS used by numba rounds one complex product inside a 3x3 zgemm.

The u1 term M_a diag(d, 0, 0) M_b has a single non-zero product per element. Each element of
np.dot(np.dot(Ma, D), Mb) is therefore ((Ma[i,0] * d) * Mb[0,j]) evaluated by the BLAS kernel,
while the rank-one variant evaluates the same two complex products with numba's scalar
arithmetic. For random operands, the BLAS result of one product x*y (np.dot of [[x]] padded
into a 3x3 matrix with a unit diagonal partner) is compared with four exactly rounded formulas,
computed with rational arithmetic (rn = round to nearest double):

  plain    re = rn(rn(xr yr) - rn(xi yi)),  im = rn(rn(xr yi) + rn(xi yr))   (numba scalar)
  fma_1    re = rn(xr yr - rn(xi yi)),      im = rn(xr yi + rn(xi yr))       (one product fused)
  fma_2    re = rn(rn(xr yr) - xi yi),      im = rn(rn(xr yi) + xi yr)       (the other fused)
  exact    re = rn(xr yr - xi yi),          im = rn(xr yi + xi yr)
and the two mixed combinations (real part of one, imaginary part of the other).

Usage: probe_complex_product.py [n]
"""

import sys
from fractions import Fraction as Fr
import numpy as np
import numba as nb

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

@nb.njit
def blas_products(x, y):
    """x[k] * y[k] through two 3x3 zgemm, with the structure of a u1 term: A = diag-padded x."""
    out = np.empty(len(x), dtype=nb.complex128)
    A = np.zeros((3,3), dtype=nb.complex128)
    B = np.zeros((3,3), dtype=nb.complex128)
    D = np.zeros((3,3), dtype=nb.complex128)
    for k in range(len(x)):
        A[0,0] = x[k]
        D[0,0] = 1.0
        B[0,0] = y[k]
        out[k] = np.dot(np.dot(A, D), B)[0,0]
    return out

@nb.njit
def scalar_products(x, y):
    out = np.empty(len(x), dtype=nb.complex128)
    for k in range(len(x)):
        out[k] = x[k] * y[k]
    return out

def rn(q):
    return float(q)

def formulas(x, y):
    xr, xi, yr, yi = Fr(x.real), Fr(x.imag), Fr(y.real), Fr(y.imag)
    rr, ii, ri, ir = xr*yr, xi*yi, xr*yi, xi*yr
    return {
        "plain": complex(rn(Fr(rn(rr)) - Fr(rn(ii))), rn(Fr(rn(ri)) + Fr(rn(ir)))),
        "fma_1": complex(rn(rr - Fr(rn(ii))), rn(ri + Fr(rn(ir)))),
        "fma_2": complex(rn(Fr(rn(rr)) - ii), rn(Fr(rn(ri)) + ir)),
        "exact": complex(rn(rr - ii), rn(ri + ir)),
        "fma_2/fma_1": complex(rn(Fr(rn(rr)) - ii), rn(ri + Fr(rn(ir)))),
        "fma_1/fma_2": complex(rn(rr - Fr(rn(ii))), rn(Fr(rn(ri)) + ir)),
    }

rng = np.random.default_rng(1)
x = (rng.normal(size=n) + 1j*rng.normal(size=n)) * 10**rng.uniform(-3, 3, n)
y = (rng.normal(size=n) + 1j*rng.normal(size=n)) * 10**rng.uniform(-3, 3, n)
blas = blas_products(x, y)
scal = scalar_products(x, y)

print(f"{n} random complex products")
print(f"BLAS (3x3 zgemm) == numba scalar product: {np.sum(blas == scal)} / {n}")
match = {k: 0 for k in ["plain", "fma_1", "fma_2", "exact", "fma_2/fma_1", "fma_1/fma_2"]}
re_match = {k: 0 for k in match}; im_match = {k: 0 for k in match}
for k in range(n):
    f = formulas(x[k], y[k])
    for name, v in f.items():
        re_match[name] += v.real == blas[k].real
        im_match[name] += v.imag == blas[k].imag
        match[name] += v == blas[k]
    if k == 0:
        assert f["plain"] == scal[0], "scalar product is not the plain formula"
for name in match:
    print(f"BLAS == {name:11s}: both parts {match[name]:6d}, real part {re_match[name]:6d}, imaginary part {im_match[name]:6d}")
d = np.abs(blas - scal) / np.abs(scal)
print(f"relative difference BLAS vs scalar: max {d.max():.2e}, median over differing {np.median(d[d > 0]) if np.any(d > 0) else 0:.2e}")
