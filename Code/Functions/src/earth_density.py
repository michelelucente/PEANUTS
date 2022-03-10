#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 10 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

import numpy as np
from math import cos, sin

import src.files as f

class EarthDensity:
  """
  See hep-ph/9702343 for the definition of trajectory coordinate and Earth density parametrisation.
  """

  def __init__(self, density_file="./Data/Earth_Density.csv"):
    """
    Read the Earth density parametrisation, in units of mol/cm^3, following hep-ph/9702343
    """

    # TODO: This is specific for the example file, make general
    earth_density = f.read_csv(density_file, names=["rj", "alpha", "beta", "gamma"], skiprows=3)

    self.rj = earth_density.rj.to_numpy()
    self.alpha = earth_density.alpha.to_numpy()
    self.beta = earth_density.beta.to_numpy()
    self.gamma = earth_density.gamma.to_numpy()

  def parameters(self, eta):
    """
    Returns the values of the density parameters for the shells crossed by the path 
    with nadir angle = eta as a list of lists, where each element [[a, b, c], x_i] refers to a 
    Earth shell  (from inner to outer layers) having density profile n_e(x) = a + b x^2 + c x^4, 
    with shell external boundary at x == x_i.
    """

    # Select the index "idx_shells" in rj such that for i >= idx_shells => rj[i] > sin(eta)
    # The shells having rj[i] > sin(eta) are the ones crossed by a path with nadir angle = eta
    idx_shells = np.searchsorted(self.rj, sin(eta))

    # Keep only the parameters for the shells crossed by the path with nadir angle eta
    alpha_prime = self.alpha[idx_shells::] + self.beta[idx_shells::] * sin(eta)**2 + self.gamma[idx_shells::] * sin(eta)**4
    beta_prime = self.beta[idx_shells::] + 2 * self.gamma[idx_shells::] * sin(eta)**2
    gamma_prime = self.gamma[idx_shells::]

    # Compute the value of the trajectory coordinates xj at each shell crossing
    xj = np.sqrt( (self.rj[idx_shells::])**2 - sin(eta)**2 )

    return [ [ [alpha_prime[i], beta_prime[i], gamma_prime[i]], xj[i] ] for i in range(len(alpha_prime))]


  def __call__(self, x, eta):
    """
    Computes the value of Earth electron density in units of mol/cm^3 for trajectory coordinate 
    x and nadir angle eta;
    """

    # The density profile is symmetric with respect to x=0 
    x = np.abs(x)

    # If x > cos(eta) the trajectory coordinate is beyond Earth surface, thus density is zero.
    if x > cos(eta):
      return 0

    # Get the parameters
    param = self.parameters(eta)
    alpha_prime = [x[0][0] for x in param]
    beta_prime = [x[0][1] for x in param]
    gamma_prime = [x[0][2] for x in param]
    xj = [x[1] for x in param]

    # The index "idx" determines within which shell xj[idx] the point x is
    idx = np.searchsorted(xj, x)

    return alpha_prime[idx] + beta_prime[idx] * x**2 + gamma_prime[idx] * x**4
