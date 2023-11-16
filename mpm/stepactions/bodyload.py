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
# Created on Thu Nov 15 13:15:14 2018

# @author: Matthias Neuner
"""
Simple body force load.
If not modified in subsequent steps, the load held constant.
"""

documentation = {
    "forceVector": "The force vector",
    "delta": "In subsequent steps only: define the updated force vector incrementally",
    "f(t)": "(Optional) define an amplitude in the step progress interval [0...1]",
}

from fe.stepactions.base.stepactionbase import StepActionBase
import numpy as np
import sympy as sp


class BodyLoad(StepActionBase):
    def __init__(self, name, model, journal, cells, bodyLoadType: str, loadVector, **kwargs):
        self.name = name

        self._loadVector = loadVector
        self._loadAtStepStart = np.zeros_like(self._loadVector)
        self.loadType = bodyLoadType
        self.cells = cells

        if len(self._loadVector) < model.domainSize:
            raise Exception("BodyForce {:}: load vector has wrong dimension!".format(self.name))

        self._delta = self._loadVector
        if "f_t" in kwargs:
            t = sp.symbols("t")
            self._amplitude = sp.lambdify(t, sp.sympify(kwargs["f_t)"]), "numpy")
        else:
            self._amplitude = lambda x: x

        self._idle = False

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

    # def updateStepAction(self, action, jobInfo, model, fieldOutputController, journal):
    #     if "forceVector" in action:
    #         self._delta = np.fromstring(action["forceVector"], sep=",", dtype=np.double) - self._loadAtStepStart
    #     elif "delta" in action:
    #         self._delta = np.fromstring(action["delta"], sep=",", dtype=np.double)

    #     if "f(t)" in action:
    #         t = sp.symbols("t")
    #         self._amplitude = sp.lambdify(t, sp.sympify(action["f(t)"]), "numpy")
    #     else:
    #         self._amplitude = lambda x: x

    #     self.idle = False

    def getCurrentBodyLoad(self, stepProgress):
        if self._idle == True:
            t = 1.0
        else:
            # incNumber, incrementSize, stepProgress, dT, stepTime, totalTime = increment
            t = stepProgress

        return self._loadAtStepStart + self._delta * self._amplitude(t)
