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

from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmpm.numerics.predictors.basepredictor import BasePredictor


class LinearPredictor(BasePredictor):
    """A linear extrapolator which uses the last solution increment to predict the next solution."""

    def __init__(self):
        self._dU_n = None
        self._deltaT_n = None

    def resetHistory(
        self,
    ):
        self._dU_n = None
        self._deltaT_n = None

    def getPrediction(self, timeStep: TimeStep):
        if self._dU_n is None:
            return None
        if timeStep.timeIncrement < 1e-15 or self._deltaT_n < 1e-15:
            return None
        else:
            dU = self._dU_n * (timeStep.timeIncrement / self._deltaT_n)
            return dU

    def updateHistory(self, dU: DofVector, timeStep: TimeStep):
        self._dU_n = dU.copy()
        self._deltaT_n = timeStep.timeIncrement
