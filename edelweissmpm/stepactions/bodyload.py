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
import sympy as sp
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmpm.sets.cellset import CellSet
from edelweissmpm.stepactions.base.mpmbodyloadbase import MPMBodyLoadBase


class BodyLoad(MPMBodyLoadBase):
    def __init__(self, name, model, journal, cells, bodyLoadType: str, loadVector, **kwargs):
        self.name = name

        self._loadVector = loadVector
        self._loadAtStepStart = np.zeros_like(self._loadVector)
        self._loadType = bodyLoadType
        self._cells = cells

        if len(self._loadVector) < model.domainSize:
            raise Exception("BodyForce {:}: load vector has wrong dimension!".format(self.name))

        self._delta = self._loadVector
        if "f_t" in kwargs:
            t = sp.symbols("t")
            self._amplitude = sp.lambdify(t, sp.sympify(kwargs["f_t)"]), "numpy")
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
            if stepMagnitude == None:
                # standard case
                self._loadAtStepStart += self._delta * self._amplitude(1.0)
            else:
                # set the 'actual' increment manually, e.g. for arc length method
                self._loadAtStepStart += self._delta * stepMagnitude

            self._delta = 0
            self._idle = True

    def getCurrentLoad(self, timeStep: TimeStep):
        if self._idle == True:
            t = 1.0
        else:
            t = timeStep.stepProgress

        return self._loadAtStepStart + self._delta * self._amplitude(t)
