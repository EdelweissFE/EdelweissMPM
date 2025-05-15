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


from collections.abc import Callable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.timesteppers.timestep import TimeStep


class Dirichlet(DirichletBase):

    def __init__(self, name, nSet, field, values, model, journal, f_t: Callable[[float], float] = None):
        """
        This is a classical dirichlet boundary condition for MPM models.

        Parameters
        ----------
        name : str
            Name of the distributed load.
        nSet : NodeSet
            The node set to apply the BC to.
        field : str
            The field to apply the BC to.

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
        f_t : Callable[[float], float]
                The amplitude function of the distributed load.
        """
        self.name = name
        self._journal = journal
        self._domainSize = model.domainSize

        self.field = field

        self.nSet = nSet
        self.fieldSize = getFieldSize(self.field, self._domainSize)

        self.updateStepAction(values, f_t)

    @property
    def components(self) -> np.ndarray:
        return self._components

    def applyAtStepEnd(self, model):
        self.active = False

    def updateStepAction(self, values, f_t: Callable[[float], float] = None):
        self.active = True

        self._components = np.array([i for i in values.keys()])

        self._delta = np.array([values for values in values.values()])

        if f_t is not None:
            self._amplitude = f_t
        else:
            self._amplitude = lambda x: x

    def getDelta(self, timeStep: TimeStep, nodes):
        if self.active:
            delta = np.tile(self._delta, len(nodes))

            return delta * (
                self._amplitude(timeStep.stepProgress)
                - (self._amplitude(timeStep.stepProgress - timeStep.stepProgressIncrement))
            )
        else:
            return np.tile(np.zeros_like(self._delta), len(nodes))
