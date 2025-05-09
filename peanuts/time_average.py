#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:39:40 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import numpy as np
from math import sin, cos, pi, ceil, floor
from cmath import asin as casin
from cmath import tan as ctan
from cmath import sin as csin
from cmath import cos as ccos
from cmath import sqrt as csqrt
from mpmath import fp, ellipf, sec, csc
from scipy import integrate
from scipy.interpolate import interp1d

import peanuts.files as f
from peanuts.utils import intersection


# Sometimes the function cmath.sqrt takes the "wrong" side of the branch cut, if its argument has vanishing
# imaginary part prepended by a negative sign. To prevent this we define a custom version of the function
def safe_csqrt(z):
    x = z.real
    y = z.imag

    if (y == 0.0) or (y == 0):
        return csqrt(x)
    else:
        return csqrt(x + z*1j)


def IndefiniteIntegralDay (T, eta, lam):
    """
    IndefiniteIntegralDay(T, eta, lam) computes the indefinite integral
    \int dT [ (1/sqrt(1 - T^2)) * (sec(lam) sin(eta)) / sqrt(1 + sec(lam) (-sec(lam) (cos(eta)^2 + T^2 sin(i)^2) +
     2 cos(eta) T sin(i) ctan(lam))) ].
     This is required in the change of variables from hourly and daily times (\tau_h, \tau_d) to nadir angle eta,
     when computing the eta exposure for a detector within a definite amount of time:
     - T:  must be interpreted as cos(\tau_d);
     - eta: the nadir angle;
     - lam: the detector latitude.
    See Appendix C in hep-ph/9702343 for a more extensive discussion.
    """

    # Earth rotation axis inclination
    i = 0.4091

    return (-2*safe_csqrt(2)*(-1 + T)**2*fp.csc(i)*fp.ellipf(casin(safe_csqrt(-(((1 + T)*(csin(i) - csin(eta + lam)))/
                                                               ((-1 + T)*(csin(i) + csin(eta + lam)))))),
    ((csin(i) + csin(eta - lam))*(csin(i) + csin(eta + lam)))/((csin(i) - csin(eta - lam))*(csin(i) - csin(eta + lam))))
            *fp.sec(lam)*csin(eta)*safe_csqrt((T*csin(i) + csin(eta - lam))/((-1 + T)*(csin(i) - csin(eta - lam))))*
            safe_csqrt(-(((1 + T)*(csin(i) - csin(eta + lam)))/((-1 + T)*(csin(i) + csin(eta + lam)))))*
            safe_csqrt((T*csin(i) - csin(eta + lam))/((-1 + T)*(csin(i) + csin(eta + lam))))*
            (csin(i) + csin(eta + lam))*safe_csqrt(-1 + ccos(eta)**2*fp.sec(lam)**2 +
        T**2*fp.sec(lam)**2*csin(i)**2 - 2*T*ccos(eta)*fp.sec(lam)*csin(i)*ctan(lam)))/((-1 + fp.csc(i)*csin(eta + lam))*
        safe_csqrt(-2 + (1 + T**2 - T**2*ccos(2*i) + ccos(2*eta))*fp.sec(lam)**2 - 4*T*ccos(eta)*fp.sec(lam)*csin(i)*ctan(lam))
        *safe_csqrt((-1 + T**2)*(-1 + ccos(eta)**2*fp.sec(lam)**2 + T**2*fp.sec(lam)**2*csin(i)**2 -
                            2*T*ccos(eta)*fp.sec(lam)*csin(i)*ctan(lam))))


def IntegralAngle (eta, lam, a1=0, a2=pi, eps=1e-5):
    """
    IntegralAngle(eta, lam, a1, a2, eps) computes the definite integral
    \int_cos(a2)^cos(a1) dT [ (1/sqrt(1 - T^2)) * (sec(lam) sin(eta)) /
    sqrt(1 + sec(lam) (-sec(lam) (cos(eta)^2 + T^2 sin(i)^2) + 2 cos(eta) T sin(i) ctan(lam))) ].
    This is required when computing the eta exposure for a detector over a finite amount of time.
    The time is expressed in radians of the Earth orbit, with origin at the winter solstice.
    Only half-orbit needs to be considered due to symmetry:
    - eta: the nadir angle;
    - lam: the detector latitude;
    - a1, a2: the starting the ending angles of the time interval. They must be comprised between 0 and pi;
    - eps: a small quantity needed to regularise the integral, which is locally divergent at extreme values
      of allowed integration range.
    """

    # Earth rotation axis inclination
    i = 0.4091

    # Check correct range of input times
    if (not 0 <= a1 <= pi) or (not 0 <= a2 <= pi) or (a1 > a2):
        raise ValueError('a1 and a2 must be comprised between 0 and pi, and a2 must be greater than a1')

    # Define intervals of valid integration
    int1 = [-1 + eps, 1 - eps] # Where the function T = cos(ai) is defined
    int2 = [sin(lam - eta)/sin(i) + eps, sin(lam + eta)/sin(i) - eps] # Range where T = cos(ai) ctan take values for fixed values of lam, eta, i
    int3 = [cos(a2), cos(a1)] # Interval of detector time exposure

    # The integration interval is given by the intersection of int1, int2, int3
    int_full = intersection(int1, int2, int3)

    # If no intersection then the integral is zero
    if len(int_full) == 0:
        return 0

    # If non-empty intersection then define the lower and upper limits of integration
    elif len(int_full) == 2:

        low, up = int_full[0], int_full[1]

        # If the integration interval is larger than 2 * eps then compute the definite integral
        # Discard the small imaginary part due to numerics
        if up - low > 2 * eps:
            return ( IndefiniteIntegralDay(up, eta, lam) - IndefiniteIntegralDay(low, eta, lam) ).real

        # Otherwise return zero
        else:
            return 0

    # The code only works  if the intersection results in a continous interval (i.e. len(int_full) == 1),
    # as is expected. If this is not the case raise an error.
    else:
        raise Exception("Unable to treat disconnected integration intervals")



def IntegralDay (eta, lam, d1=0, d2=365):
    """
    IntegralDay(eta, lam, d1, d2) computes the non-normalised exposure on the nadir angle eta for an
    experiment located at latitude lam (in radians), taking data from day d1 to day d2.
    The time origin day = 0 is the northern hemisphere winter solstice midnight.
    The function accepts values of d1, d2 comprised between zero and 365.
    - eta: the nadir angle;
    - lam: the detector latitude;
    - d1: lower limit of day interval
    - d2: upper limit of day interval
    """

    # Check correct range of input times
    if (not 0 <= d1 <= 365) or (not 0 <= d2 <= 365) or (d1 > d2):
        raise ValueError('d1 and d2 must be comprised between 0 and 365, and d2 must be greater than d1')

    # Convert days to angles
    [a1, a2] = 2 * pi * np.array([d1, d2]) / 365

    # The calculation is performed separately for first and second half of the year
    int1 = intersection([a1, a2], [0, pi]) # Between winter and summer solstices
    int2 = intersection([a1, a2], [pi, 2 * pi]) # Between summer and winter solstices

    # Compute the exposure for each half-year
    weight1 = IntegralAngle(eta, lam, int1[0], int1[1]) if len(int1) == 2 else 0

    # For the second half we use the symmetry of the orbit to recast the trajectory into an equivalent one
    # in the range of days between 0 and 365/2.
    weight2 = IntegralAngle(eta, lam, 2*pi - int2[1], 2*pi - int2[0]) if len(int2) == 2 else 0

    # Return the sum of the exposures
    return weight1 + weight2


def NadirExposure(lam=-1, d1=0, d2=365, ns=1000, normalized=False, from_file=None, angle="Nadir", daynight=None):
    """
    NadirExposure(lam, d1, d2, ns) computes the exposure for ns nadir angle samples
    for an experiment located at latitude lam (in radians), taking data from day d1 to day d2.
    - lam: the latitude of the experiment (def. -1)
    - d1: lower limit of day interval
    - d2: upper limit of day interval
    - ns: number of nadir angle samples
    - normalized: normalization of exposure
    - from_file: file with experiments exposure
    - angle: angle of samples is exposure file
    """

    # Generate ns samples of the nadir angle between 0 and pi
    eta_samples = np.linspace(0, pi, ns)
    if daynight == "day":
      eta_samples = eta_samples[ceil(ns/2):]
    elif daynight == "night":
      eta_samples = eta_samples[:floor(ns/2)]
    deta = eta_samples[1]-eta_samples[0]

    # Get exposure from file
    if from_file is not None:
      raw_exposure=f.read_csv(from_file, skiprows=9, names=['Exposure'])

      if ns != len(raw_exposure['Exposure']):
        print("Error: number of samples must match that in the data file")
        exit()

      # Rearrange exposure values by the angle they use
      if angle == "Nadir":
        exposure = raw_exposure["Exposure"]
      # Zenith = pi - Nadir, Zenith = [0,pi]
      if angle == "Zenith":
        exposure = raw_exposure["Exposure"].reverse()
      # CosZenith = cos(pi - Nadir), CosZenith = [-1,1]
      if angle == "CosZenith":
        cz_samples = np.linspace(-1,1,ns)
        dcz = cz_samples[1]-cz_samples[0]
        exposure_interp = interp1d(cz_samples, raw_exposure["Exposure"], kind='cubic')
        exposure = [exposure_interp(-cos(eta_samples[i]))*sin(eta_samples[i])*deta/dcz for i in range(len(eta_samples))]
        exposure = [exp if exp > 0 else 0 for exp in exposure]

    elif lam >= 0:
      # Compute exposure integrating in the given time ranges
      exposure = np.array([IntegralDay(eta, lam, d1, d2) for eta in eta_samples])

    else:
      # If there is not file, there must be a latitude
      print("Error: to compute the integrated probability either the latitude or a exposure file is needed, please provide either.")
      exit()

    # Normalize the distribution if requested
    if normalized:
        norm = integrate.trapz(x=eta_samples,y=exposure)
        exposure = exposure/norm

    return np.vstack((eta_samples, exposure)).T
