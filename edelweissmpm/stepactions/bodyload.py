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
import numpy as np
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmpm.sets.cellset import CellSet
from edelweissmpm.stepactions.base.mpmbodyloadbase import MPMBodyLoadBase


class BodyLoad(MPMBodyLoadBase):
    def __init__(self, name, model, journal, cells, bodyLoadType: str, loadVector, **kwargs):
        """
        This is a classical body load for MPM models.

        Parameters
        ----------
        name : str
            Name of the distributed load.
        model : MPMModel
            The MPM model tree.
        journal : Journal
            The journal to write messages to.
        cells: CellSet
            The cells to apply the distributed load to.
        bodyLoadType: str
            The type of the body load, e.g., "gravity".
        loadVector : np.ndarray
            The load vector to apply to the particles.
        **kwargs
            Additional keyword arguments. The following are supported:
            - f_t : Callable[[float], float]
                The amplitude function of the distributed load.
        """
        self.name = name

        self._loadVector = loadVector
        self._loadAtStepStart = np.zeros_like(self._loadVector)
        self._loadType = bodyLoadType
        self._cells = cells

        if len(self._loadVector) < model.domainSize:
            raise Exception("BodyForce {:}: load vector has wrong dimension!".format(self.name))

        self._delta = self._loadVector
        if "f_t" in kwargs:
            self._amplitude = kwargs["f_t"]
        else:
            self._amplitude = lambda x: x

        self._idle = False

    @property
    def cellSet(self) -> CellSet:
        return self._cells

    @property
    def loadType(self) -> str:
        return self._loadType

    def applyAtStepEnd(self, model, stepMagnitude=None):
        if not self._idle:
            if stepMagnitude is None:
                # standard case
                self._loadAtStepStart += self._delta * self._amplitude(1.0)
            else:
                # set the 'actual' increment manually, e.g. for arc length method
                self._loadAtStepStart += self._delta * stepMagnitude

            self._delta = 0
            self._idle = True

    def getCurrentLoad(self, timeStep: TimeStep):
        if self._idle:
            t = 1.0
        else:
            t = timeStep.stepProgress

        return self._loadAtStepStart + self._delta * self._amplitude(t)
