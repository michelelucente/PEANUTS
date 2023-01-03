#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 3 2022

@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

with_slha = True
try:
  import pyslha # pyslha module for reading SLHA files, by Andy Buckley (https://arxiv.org/abs/1305.4194)
except:
  print("Warning!: Python module pyslha not found, disabling slha reading/writing routines")
  with_slha = False


import pandas as pd
import yaml
from decimal import Decimal

from src.settings import Settings
import os


def read_csv(*args, **kwargs):
  """
  Function to read csv. Just a wrapper around the pandas version
  """
  return pd.read_csv(*args, **kwargs)

def read_yaml(filepath):
  """
  Function to read from yaml files
  """
  settings = Settings(yaml.safe_load(open(filepath)))

  return settings

def read_slha(filepath):
  """
  Function to read from slha files
  """

  if not with_slha:
    print("Error!: Tried to use pyslha but module is not installed")
    exit()


  slha = pyslha.read(filepath)
  blocks = [''.join(k) for k in slha.blocks.keys()]

  # Check that the slha contains the important blocks
  if 'SMINPUTS' not in blocks:
    raise Exception("Error: SLHA file has the wrong format, the block SMINPUTS in missing")
  if 'UPMNSIN' not in blocks:
    raise Exception("Error: SLHA file has the wrong format, the block UPMNS in missing")

  # Extract masses from the SMINPUTS block (converting from GeV to eV)
  mNu1 = slha.blocks['SMINPUTS'][12] * 1e9
  mNu2 = slha.blocks['SMINPUTS'][14] * 1e9
  mNu3 = slha.blocks['SMINPUTS'][8] * 1e9

  # Extract PMNS angles and phases from the UPMNSIN block
  theta12 = slha.blocks['UPMNSIN'][1]
  theta23 = slha.blocks['UPMNSIN'][2]
  theta13 = slha.blocks['UPMNSIN'][3]
  delta = slha.blocks['UPMNSIN'][4]
  # TODO: I don't think these are useful in any way, so commented for now
  #alpha1 = slha.blocks['UMPNSIN'][5]
  #alpha2 = slha.blocks['UMPNSIN'][6]

  # Build dictionary of neutrino parameters
  nu_params = {'dm21' : mNu2**2 - mNu1**2,
               'dm31' : mNu3**2 - mNu1**2,
               'theta12'    : theta12,
               'theta23'    : theta23,
               'theta13'    : theta13,
               'delta'      : delta}

  return nu_params

def output(settings, outs):

  def dec(x):
    return "{:.5E}".format(Decimal(x))

  towrite = ""

  if settings.flux:
    towrite += "# Flux [cm^-2 s^-1]\t" + str(dec(outs[0]["flux"])) + "\n"

  if settings.probabilities:
    towrite += "\n# Probabilities\n"
    towrite += "# E [MeV]\t"
    if settings.solar:
      towrite += "Psolar (e) \t Psolar (mu) \t Psolar (tau)\t"
    if settings.earth:
      towrite += "Pearth (e) \t Pearth (mu) \t Pearth (tau)\t"
    towrite += "\n"

    for i in range(len(settings.energy)):

      towrite += str(dec(settings.energy[i])) + "\t"

      if settings.solar:

        for out in outs[i]["solar"]:
          towrite += str(dec(out)) + "\t"

      if settings.earth:
        for out in outs[i]["earth"]:
          towrite += str(dec(out)) + "\t"

      towrite += "\n"

  if settings.undistorted_spectrum:
    towrite += "\n# Spectrum (undistorted)\n"
    towrite += "# E [MeV] \t Spec (e)\n"

    for i in range(len(settings.energy)):

      towrite += str(dec(settings.energy[i])) + "\t"
      towrite += str(dec(outs[i]['spectrum']))

    towrite += "\n"


  elif settings.distorted_spectrum:
    towrite += "\n# Spectrum (distorted)\n"
    towrite += "# E [MeV] \t Spec (e) \t Spec (mu) \t Spec (tau)\n"

    for i in range(len(settings.energy)):

     towrite += str(dec(settings.energy[i])) + "\t"
     for out in outs[i]["spectrum"]:
       towrite += str(dec(out)) + "\t"

    towrite += "\n"



  if settings.output == "stdout":

    print()
    print(towrite)

  else:

    try:

      path = os.path.dirname(os.path.realpath( __file__ ))
      f = open(path + "/../" + settings.output, "w")
      f.write(towrite)
      f.close()

      print("Output written to file ", settings.output)

    except FileNotFoundError:
      print("Error: output file not found.")
      exit()



