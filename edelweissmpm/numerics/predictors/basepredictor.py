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
"""
Extrapolators (Predictors) are used for making initia guesses in the Newton scheme.
"""

from abc import ABC, abstractmethod

from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.timesteppers.timestep import TimeStep


class BasePredictor(ABC):
    """
    The BaseExtrapolator class is an abstract base class for all extrapoltors.
    If you want to implement a new extrapolator, you have to inherit from this class."""

    @abstractmethod
    def resetHistory(
        self,
    ):
        """
        Reset the history of the extrapolator. Usually when a solve fails.
        """

    @abstractmethod
    def getPrediction(self, timeStep: TimeStep) -> DofVector:
        """
        Get the prediction for the next time step.

        Returns
        -------
        DofVector
            The predicted solution for the next time step.
        """

    @abstractmethod
    def updateHistory(self, dU: DofVector, timeStep: TimeStep) -> None:
        """
        Update the history of the extrapolator.

        Parameters
        ----------
        dU
            The increment of the solution.
        timeStep
            The current time step.
        """
