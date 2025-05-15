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

from collections.abc import Callable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.numerics.dofmanager import DofManager
from edelweissmpm.particles.base.baseparticle import BaseParticle
from edelweissmpm.stepactions.base.arclengthcontrollerbase import (
    ArcLengthControllerBase,
)


class IndirectControl(ArcLengthControllerBase):
    identification = "IndirectControl"

    """This class represent an ArcLengthControllerBase compatible
    module for indirect (displacement) controlled simulations.

    Parameters
    ----------
    name
        The name for printing purposes.
    model
        The model tree instance.
    partciles

    L
        The target arc length.
    cMatrix
        The c-vector in matrix form: rows=material points, columns=c vector for this material point.
    field
        The field for which the arc length parameter is computed.
    journal
        The journal instance for logging purposes.
    f_t
        A function for the time dependent scaling of the arc length parameter.
        """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        particles: list[BaseParticle],
        L: float,
        cMatrix: np.ndarray,
        field: str,
        journal: Journal,
        f_t: Callable[[float], float] = None,
    ):
        self._name = name
        self._journal = journal
        self._L = L
        self._currentL0 = 0.0

        self._particles = particles

        self._field = field
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._c = cMatrix.flatten()

        if f_t is None:
            self._f_t = lambda t: 1.0
        else:
            self._f_t = f_t

    def computeDDLambda(
        self, dU: DofVector, ddU_0: DofVector, ddU_f: DofVector, timeStep: TimeStep, dofManager: DofManager
    ) -> float:
        """
        This method is called by an arc length solver for computing the
        correction to the arc length parameter.

        Parameters
        ----------
        dU
            The current solution increment.
        ddU_0
            The dead (unconditional) correction to the solution.
        ddU_f
            The controllable correction to the solution resulting from the reference load.
        timeStep
            The current time increment.
        dofManager
            The DofManager instance.

        Returns
        -------
        float
            The computed correction to the arc length parameter.
        """

        mpIndices = [
            np.asarray(
                [dofManager.idcsOfFieldVariablesInDofVector[kf.node.fields[self._field]] for kf in p.kernelFunctions]
            ).flatten()
            for p in self._particles
        ]

        Ns = [np.asarray(p.getInterpolationVector(p.getCenterCoordinates())).flatten() for p in self._particles]

        dT = self._f_t(timeStep.stepProgress + timeStep.stepProgressIncrement) - self._f_t(timeStep.stepProgress)
        dL = dT * (self._L - self._currentL0)

        ddUMP_f = np.asarray([N @ ddU_f[idcs].reshape((-1, self._fieldSize)) for N, idcs in zip(Ns, mpIndices)])
        ddUMP_0 = np.asarray([N @ ddU_0[idcs].reshape((-1, self._fieldSize)) for N, idcs in zip(Ns, mpIndices)])
        dUMP = np.asarray([N @ dU[idcs].reshape((-1, self._fieldSize)) for N, idcs in zip(Ns, mpIndices)])

        ddLambda = (dL - self._c.dot(dUMP.flatten() + ddUMP_0.flatten())) / self._c.dot(ddUMP_f.flatten())

        return ddLambda

    def applyAtStepEnd(self, model):
        self._currentL0 = self._L
