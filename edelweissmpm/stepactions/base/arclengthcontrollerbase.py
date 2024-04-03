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

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissmpm.numerics.dofmanager import DofManager
from edelweissfe.numerics.dofmanager import DofVector
import numpy as np
from abc import ABC, abstractmethod


class ArcLengthControllerBase(StepActionBase):
    @abstractmethod
    def computeDDLambda(
        self, dU: DofVector, ddU_0: DofVector, ddU_f: DofVector, timeStep: TimeStep, dofManager: DofManager
    ) -> float:
        """This method is response for compute the correction to dLamba, i.e., the increment of the arc length parameter.

        Parameters
        ----------
        dU
            The current increment of the solution.
        ddU_0
            The current dead increment of the solution.
        ddU_f
            The current reference increment of the solution.
        timeStep
            The current time increment.
        dofManager
            The DofManager instance.

        Returns
        -------
        float
            The correction ddLambda to dLambda.
        """
        pass
