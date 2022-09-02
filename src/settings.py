#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on My 11 2022

@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import types
import numpy as np

from src.pmns import PMNS

class Settings:

  def __init__(self, *args):

    self.solar = False
    self.earth = False

    # If there is only one argument, it is a settings dictionary
    if len(args) == 1:
      settings = args[0]

      # Select mode first
      if "Solar" in settings:
        self.solar = True
      if "Earth" in settings:
        self.earth = True
      if not self.solar and not self.earth:
        print("Error: unkown mode, please provide a running mode, \"Solar\", \"Earth\" or both")
        exit()

      # Extract neutrino parameters
      if "Neutrinos" not in settings:
        print("Error: missing neutrino information, please provide dm21, dm31, theta12, theta13, theta23 and delta")
        exit()
      elif isinstance(settings["Neutrinos"], dict):
        if "dm21" not in settings["Neutrinos"] or\
           "dm31" not in settings["Neutrinos"] or\
           "theta12" not in settings["Neutrinos"] or\
           "theta13" not in settings["Neutrinos"] or\
           "theta23" not in settings["Neutrinos"] or\
           "delta" not in settings["Neutrinos"] :
          print("Error: missing neutrino information, please provide dm21, dm31, theta12, theta13, theta23 and delta")
          exit()
        else:
          self.dm21 = settings["Neutrinos"]["dm21"]
          self.dm31 = settings["Neutrinos"]["dm31"]
          self.pmns = PMNS(settings["Neutrinos"]["theta12"],\
                           settings["Neutrinos"]["theta13"],\
                           settings["Neutrinos"]["theta23"],\
                           settings["Neutrinos"]["delta"])
      else:

        import src.files as f

        slha_file = settings["Neutrinos"]
        try:
          nu_params = f.read_slha(slha_file)
          th12 = nu_params['theta12']
          th13 = nu_params['theta13']
          th23 = nu_params['theta23']
          d = nu_params['delta']
          self.pmns = PMNS(th12, th13, th23, d)
          self.dm21 = nu_params['dm21']
          self.dm31 = nu_params['dm31']
        except FileNotFoundError:
          print("Error: slha file " + slha_file + " not found.")
          exit()



      # Extract solar parameters
      if "Solar" in settings:

        if "fraction" not in settings["Solar"]:
          print("Error: missing solar neutrino fraction.")
          exit()
        else:
          self.fraction = settings["Solar"]["fraction"]

        self.solar_file = settings["Solar"]["solar_model"] if "solar_model" in settings["Solar"] else None

        self.probabilities = settings["Solar"]["probabilities"] if "probabilities" in settings["Solar"] else True

        self.flux = settings["Solar"]["flux"] if "flux" in settings["Solar"] else False

        self.undistorted_spectrum = False
        self.distorted_spectrum =  False
        if "spectrum" in settings["Solar"]:
          if settings["Solar"]["spectrum"] == "undistorted":
            self.undistorted_spectrum = True
          elif settings["Solar"]["spectrum"] == "distorted":
            self.distorted_spectrum = True
          else:
            print("Error: unknown option for spectrum, select undistorted or distorted")
            exit()

      # Extract earth parameters
      if "Earth" in settings:

        if "Solar" not in settings and\
           ("state" not in settings["Earth"] or "basis" not in settings["Earth"]):
          print("Error: missing input neutrino state or basis, please provide both.")
          exit()
        elif "Solar" not in settings:
          self.nustate = np.array(settings["Earth"]["state"])
          self.basis = settings["Earth"]["basis"]

        if "depth" not in settings["Earth"]:
          print("Error: missing depth of experiment, please provide it.")
          exit()
        else:
          self.depth = settings["Earth"]["depth"]

        # Either a specific nadir angle, eta, or a latitude must be provided
        if "eta" not in settings["Earth"] and "latitude" not in settings["Earth"]:
          print("Error: missing nadir angle (eta) and latitude, please provide either.")
          exit()
        elif "eta" in settings["Earth"]:
          self.eta = settings["Earth"]["eta"]
          self.exposure = False
        elif "latitude" in settings["Earth"]:
          self.latitude = settings["Earth"]["latitude"]
          self.exposure = True
          self.exposure_normalized = settings["Earth"]["exposure_normalized"] if "exposure_normalized" in settings["Earth"] else False
          self.exposure_time = settings["Earth"]["exposure_time"] if "exposure_time" in settings["Earth"] else [0,365/2]
          self.exposure_samples = settings["Earth"]["exposure_samples"] if "exposure_samples" in settings["Earth"] else 1000
          self.exposure_file = settings["Earth"]["exposure_file"] if "exposure_file" in settings["Earth"] else None
          self.exposure_angle = settings["Earth"]["exposure_angle"] if "exposure_angle" in settings["Earth"] else "Nadir"
        else:
          print("Error: both nadir angle (eta) and latitude are provided, but only one should be.")
          exit()

        self.density_file = settings["Earth"]["density"] if "density" in settings["Earth"] else None
        self.evolution = settings["Earth"]["evolution"] if "evolution" in settings["Earth"] else "analytical"

      # Extract energy
      if "Energy" not in settings:
        print("Error: missing energy, please provide value or range.")
        exit()
      elif isinstance(settings["Energy"], list):
        # Assume it is given as [min, max] or [min, max, step]
        if len(settings["Energy"]) < 2 or len(settings["Energy"]) > 3:
          print("Error: Energy should be given as single number or as range [Emin, Emax, (Estep).")
          exit()
        Emin = settings["Energy"][0]
        Emax = settings["Energy"][1]
        if len(settings["Energy"]) == 3:
          Estep = settings["Energy"][2]
          N = int( (Emax-Emin)/Estep)+1
        else:
          # If step is not given, assume 10 iterations
          N = 10
          Estep = (Emax-Emin)/(N-1)
        self.energy = [Emin + i*Estep for i in range(0,N)]
      else:
        self.energy = [settings["Energy"]]

      # Select printing mode, default is stdout
      if "Output" in settings:
        self.output = settings["Output"]
      else:
        self.output = "stdout"

    # If there are exactly 6 arguments, we are in solar mode
    # args = (pmns, dm21, dm31, E, fraction, options)
    elif len(args) == 6:

      self.solar = True
      self.pmns = args[0]
      self.dm21 = args[1]
      self.dm31 = args[2]
      self.energy = [args[3]]
      self.fraction = args[4]
      self.solar_file = args[5].solar if args[5].solar != "" else None

    # If there are exactly 7 arguments, we are in earth mode
    # args = (pmns, dm21, dm31, E, eta, H, options)
    elif len(args) == 7:

      self.earth = True
      self.pmns = args[0]
      self.dm21 = args[1]
      self.dm31 = args[2]
      self.energy = [args[3]]
      self.eta = args[4]
      self.depth = args[5]

      if args[6].flavour is not None:
        self.nustate = args[6].flavour
        self.basis = "flavour"
      elif args[6].mass is not None:
        self.nustate = args[6].mass
        self.basis = "mass"
      else:
        print("Error: unknown basis, please choose either flavour or mass basis.")
        exit()

      self.density_file = args[6].density if args[6].density != "" else None
      self.evolution = "analytical" if args[6].analytical else "numerical"


