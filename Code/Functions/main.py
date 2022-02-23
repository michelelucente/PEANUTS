#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:47:01 2022

@author: michele
"""
import pandas as pd
import numpy as np
from numpy import arctan, arcsin
from probability_sun_energy import PnuenueReaction
from earth_density import EarthDensity


# Import data from solar model
solar_model = pd.read_csv("./Data/bs2005agsopflux.csv", 
                          usecols=[1, 3, 7, 13],
                          names = ['radius', 'density_log_10', '8B fraction', 'hep fraction'],
                          sep=" ", skiprows=27, header=None)

solar_model['density'] = 10**solar_model['density_log_10']


# Define paths to save plots
from pathlib import Path
project_folder = str(Path(Path.cwd()).parents[1])
plots_folder = project_folder + "/TeX/figs/"


# Test Sun survival probabilities
from math import sqrt
import matplotlib.pyplot as plt

th12 = arctan(sqrt(0.469))
th13 = arcsin(sqrt(0.01))
DeltamSq21 = 7.9e-5
DeltamSq31 = 2.46e-3
E = 10
ne = 100
radius_samples = solar_model.radius
density = solar_model.density
fraction = solar_model['8B fraction']

xrange = np.arange(1,20,0.1)
ProbB8 = [PnuenueReaction(th12, th13, DeltamSq21, DeltamSq31, E, radius_samples, density, solar_model['8B fraction']) for E in xrange]
Probhep = [PnuenueReaction(th12, th13, DeltamSq21, DeltamSq31, E, radius_samples, density, solar_model['hep fraction']) for E in xrange]

SNO_B8 = pd.read_csv("./Data/B8.csv", names=['energy', 'Pnuenue'])
SNO_hep = pd.read_csv("./Data/hep.csv", names=['energy', 'Pnuenue'])


plt.plot(xrange, ProbB8, label="This code")
plt.plot(SNO_B8.energy, SNO_B8.Pnuenue, label='SNO_example')
plt.title("${}^8$B neutrinos")
plt.xlabel('Energy [MeV]')
plt.ylabel(r"$P_{\nu_e \rightarrow \nu_e}$")
plt.legend()
plt.savefig(plots_folder + "8B_SNO_cmparison.pdf")

plt.show()


plt.plot(xrange, Probhep, label="This code")
plt.plot(SNO_hep.energy, SNO_hep.Pnuenue, label='SNO_example')
plt.title("hep neutrinos")
plt.xlabel('Energy [MeV]')
plt.ylabel(r"$P_{\nu_e \rightarrow \nu_e}$")
plt.legend()
plt.savefig(plots_folder + "hep_SNO_comparison.pdf")

plt.show()


# Test Earth density profiles
from math import pi

x = np.arange(0,1,0.001)
eta = [0, pi/6, pi/4, pi/3]
labels = ["0", "pi/6", "pi/4", "pi/3"]

density = [ [EarthDensity(r, n) for r in x] for n in eta]

plt.xlabel("Nadir angle $\eta$")
plt.ylabel("Density [mol/cm${}^3$]")
for i in range(len(density)):
    plt.plot(x,density[i], label = "$\eta$ = %s" %labels[i])
plt.legend()
plt.savefig(plots_folder + "earth_density.pdf")
    
plt.show()



# Test Earth regeneration
from pmns import R12, R13, R23, Delta
from numpy.linalg import multi_dot
from math import sin, cos
from scipy.integrate import complex_ode

from potentials import k, MatterPotential, R_E
from evolutor import FullEvolutor

DeltamSq21 = 7.42e-5
DeltamSq31 = 2.514e-3
[th12, th13, th23, d] = [0.583638, 0.149575, 0.855211, 3.40339]

r13 = R13(th13)
r12 = R12(th12)
r23 = R23(th23)
delta = Delta(d)

pmns = multi_dot([r23, delta, r13, delta.conjugate(), r12])

U = np.dot(r13, r12)

# Case 1: 0 <= eta <= pi/2
eta = np.random.uniform(0, pi/2)
E = np.random.uniform(1,20)

Hk = multi_dot([U, np.diag(k(np.array([0, DeltamSq21, DeltamSq31]), E)), U.transpose()])

H = 2e3 # meters
h = H/R_E
r_d = 1 - h
x_d = sqrt(r_d**2 - sin(eta)**2)
Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)
n_1 = EarthDensity(x=1 - h/2)

params = EarthDensity(eta=eta, parameters=True)
x1, x2 = (-params[-1][1], x_d) if 0 <= eta < pi/2 else (0, Deltax)

def model(t, y):
    nue, numu, nutau = y
    dnudt = - 1j * np.dot(multi_dot([r23, delta.conjugate(), Hk + np.diag([
        MatterPotential(EarthDensity(t, eta=eta)) if 0 <= eta < pi/2 else n_1
        ,0,0]), delta, r23.transpose()]), [nue, numu, nutau])
    return dnudt

nu0 = (pmns.transpose()[1, :]).conjugate()

nu = complex_ode(model)

nu.set_integrator("Isoda")
nu.set_initial_value(nu0, x1)


x = np.linspace(x1, x2, 10**3)
sol = [nu.integrate(xi) for xi in x[1::]]
sol.insert(0, np.array(nu0)[0])

One_num = sol[-1]
One_an = np.dot(FullEvolutor(0, DeltamSq21, DeltamSq31, E, th12, th13, th23, d, eta, H), nu0.transpose()).transpose()

err = np.linalg.norm(One_num - One_an)/np.linalg.norm(One_num + One_an)

print("For E = %.2f and eta = %.2f pi the relative error between analytic and numerical solutions is %f" % (E, eta/pi, err))


# Case 2: pi/2 <= eta <= pi
eta = np.random.uniform(pi/2, pi)
E = np.random.uniform(1,20)

Hk = multi_dot([U, np.diag(k(np.array([0, DeltamSq21, DeltamSq31]), E)), U.transpose()])

x_d = sqrt(r_d**2 - sin(eta)**2)
Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)

params = EarthDensity(eta=eta, parameters=True)
x1, x2 = (-params[-1][1], x_d) if 0 <= eta < pi/2 else (0, Deltax)

def model(t, y):
    nue, numu, nutau = y
    dnudt = - 1j * np.dot(multi_dot([r23, delta.conjugate(), Hk + np.diag([
        MatterPotential(EarthDensity(t, eta=eta)) if 0 <= eta < pi/2 else n_1
        ,0,0]), delta, r23.transpose()]), [nue, numu, nutau])
    return dnudt

nu0 = (pmns.transpose()[1, :]).conjugate()

nu = complex_ode(model)

nu.set_integrator("Isoda")
nu.set_initial_value(nu0, x1)


x = np.linspace(x1, x2, 10**3)
sol = [nu.integrate(xi) for xi in x[1::]]
sol.insert(0, np.array(nu0)[0])

One_num = sol[-1]
One_an = np.dot(FullEvolutor(0, DeltamSq21, DeltamSq31, E, th12, th13, th23, d, eta, H), nu0.transpose()).transpose()

err = np.linalg.norm(One_num - One_an)/np.linalg.norm(One_num + One_an)

print("For E = %.2f and eta = %.2f pi the relative error between analytic and numerical solutions is %f" % (E, eta/pi, err))