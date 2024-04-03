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


from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
import numpy as np
import sympy as sp


class Dirichlet(DirichletBase):
    """Dirichlet boundary condition, based on a node set"""

    def __init__(self, name, nSet, field, values, model, journal, **kwargs):
        self.name = name
        self._journal = journal
        self._domainSize = model.domainSize

        self.field = field

        self.nSet = nSet
        self.fieldSize = getFieldSize(self.field, self._domainSize)

        self.updateStepAction(values, **kwargs)

    @property
    def components(self) -> np.ndarray:
        return self._components

    def applyAtStepEnd(self, model):
        self.active = False

    def updateStepAction(self, values, **kwargs):
        self.active = True

        self._components = np.array([i for i in values.keys()])

        self._delta = np.array([values for values in values.values()])

        self._amplitude = self._getAmplitude(**kwargs)

    def getDelta(self, timeStep: TimeStep, nodes):
        if self.active:
            delta = np.tile(self._delta, len(nodes))

            return delta * (
                self._amplitude(timeStep.stepProgress)
                - (self._amplitude(timeStep.stepProgress - timeStep.stepProgressIncrement))
            )
        else:
            return np.tile(np.zeros_like(self._delta), len(nodes))

    def _getAmplitude(self, **kwargs: dict) -> callable:
        """Determine the amplitude for the step, depending on a potentially specified function.

        Parameters
        ----------
        action
            The dictionary defining this step action.

        Returns
        -------
        callable
            The function defining the amplitude depending on the step propress.
        """

        if "f_t" in kwargs:
            t = sp.symbols("t")
            amplitude = sp.lambdify(t, sp.sympify(kwargs["f_t"]), "numpy")
        else:
            amplitude = lambda x: x

        return amplitude
