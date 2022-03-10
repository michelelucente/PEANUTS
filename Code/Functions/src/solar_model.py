#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 7 2022

@author Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

import src.files as f

class SolarModel:
    """"
    Class containing the info of the solar model
    """

    def __init__(self, filename="./Data/bs2005agsopflux.csv"):
        """
        Constructor of the solar model. 
        Reads the solar model file and fills useful variables
        """

        # Set file name
        self.filename = "./Data/bs2005agsopflux.csv" if filename == "" else filename
          
        # Import data from solar model
        # TODO: This assumes that any solar model file is the same format, make it more general
        self.model = f.read_csv(filename, 
                                usecols=[1, 3, 7, 13],
                                names = ['radius', 'density_log_10', '8B fraction', 'hep fraction'],
                                sep=" ", skiprows=27, header=None)

        # Set useful variables
        self.radius = self.model['radius']
        self.density =  10**self.model['density_log_10']
        self.fraction = {'8B' : self.model['8B fraction'],
                         'hep': self.model['hep fraction']}

    def radius(self):
        """
        Returns the radius column of the solar model
        """

        return self.radius

    def density(self):
        """
        Returns the density column of the solar model
        """

        return self.density



    def fraction(self, name):
        """
        Returns the fraction of neutrinos for the requested column
        """

        return self.fraction[name]

    def has_fraction(self, name):
       """
       Returns whether the solar model contains the given neutrino sample fraction
       """

       return name in self.fraction.keys()
