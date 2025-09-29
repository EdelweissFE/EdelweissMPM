# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissMPM.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMPM.
#  ---------------------------------------------------------------------
"""Body loads are loads that act on the entire body of the model."""


from abc import abstractmethod

import numpy as np
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmpm.sets.cellset import CellSet


class MPMBodyLoadBase(StepActionBase):
    """Base class for body loads.

    If you want to implement a new body load, you should derive from this class.
    """

    @property
    @abstractmethod
    def cellSet(self) -> CellSet:
        """The cells this load is acting on.

        Returns
        -------
        CellSet
            The list of cells.
        """

    @property
    @abstractmethod
    def loadType(self) -> str:
        """Return the type of load (e.g., bodyforce, ... )

        Returns
        -------
        str
            The load type.
        """

    @abstractmethod
    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """Return the current magnitude for this body load.

        Parameters
        ----------
        timeStep
            The current time step.

        Returns
        -------
        np.ndarray
            The magnitude.
        """
