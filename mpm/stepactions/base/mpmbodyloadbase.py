#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

from fe.stepactions.base.stepactionbase import StepActionBase
from fe.timesteppers.timestep import TimeStep
from mpm.sets.cellset import CellSet
from fe.config.phenomena import getFieldSize
import numpy as np
import sympy as sp
from abc import ABC, abstractmethod


class MPMBodyLoadBase(StepActionBase):
    @property
    @abstractmethod
    def cellSet(self) -> CellSet:
        """The cells this load is acting on.

        Returns
        -------
        CellSet
            The list of cells.
        """
        pass

    @property
    @abstractmethod
    def loadType(self) -> str:
        """Return the type of load (e.g., bodyforce, ... )

        Returns
        -------
        float
            The magnituide.
        """
        pass

    @abstractmethod
    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """Return the current magnitude for this body load.

        Returns
        -------
        np.ndarray
            The magnitude.
        """
        pass
