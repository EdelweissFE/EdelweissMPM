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
from fe.utils.exceptions import (
    ReachedMaxIncrements,
    ReachedMaxIterations,
    ReachedMinIncrementSize,
    DivergingSolution,
    ConditionalStop,
    StepFailed,
)

from collections import defaultdict
from fe.journal.journal import Journal
from fe.config.linsolve import getLinSolverByName, getDefaultLinSolver
from fe.config.timing import createTimingDict
from fe.config.phenomena import getFieldSize
from fe.numerics.csrgenerator import CSRGenerator
from mpm.models.mpmmodel import MPMModel
from fe.stepactions.base.stepactionbase import StepActionBase
from mpm.stepactions.base.mpmbodyloadbase import MPMBodyLoadBase
from mpm.stepactions.base.mpmdistributedloadbase import MPMDistributedLoadBase
from fe.timesteppers.timestep import TimeStep
from fe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from fe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from fe.utils.fieldoutput import FieldOutputController
from fe.constraints.base.constraintbase import ConstraintBase

from mpm.stepactions.base.mpmbodyloadbase import MPMBodyLoadBase
from mpm.stepactions.base.mpmdistributedloadbase import MPMDistributedLoadBase
from fe.stepactions.base.dirichletbase import DirichletBase
from mpm.stepactions.base.arclengthcontrollerbase import ArcLengthControllerBase
from mpm.solvers.nqsmarmotparallel import NQSParallelForMarmot

from mpm.fields.nodefield import MPMNodeField
from mpm.numerics.dofmanager import MPMDofManager
from mpm.models.mpmmodel import MPMModel
from fe.sets.nodeset import NodeSet
from scipy.sparse import csr_matrix
from numpy import ndarray
import fe.utils.performancetiming as performancetiming
import traceback


class NonlinearQuasistaticMarmotArcLengthSolver(NQSParallelForMarmot):
    """This is the serial nonlinear implicit quasi static solver.


    Parameters
    ----------
    journal
        The journal instance for logging.
    """

    identification = "MPM-NQS-ArcLength"

    def solveStep(
        self,
        timeStepper,
        linearSolver,
        mpmManager,
        dirichlets,
        bodyLoads: list[MPMBodyLoadBase],
        distributedLoads: list[MPMDistributedLoadBase],
        constraints,
        model: MPMModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: list[OutputManagerBase],
        userIterationOptions: dict,
        arcLengthController: ArcLengthControllerBase,
    ) -> tuple[bool, MPMModel]:
        """Public interface to solve for a step.

        Parameters
        ----------
        timeStepper
            The timeStepper instance.
        linearSolver
            The linear solver instance to be used.
        mpmManager
            The MPMManagerBase instance to be used for updating the connectivity.
        dirichlets
            The list of dirichlet StepActions.
        bodyLoads
            The list of bodyload StepActions.
        constraints
            The list of constraints.
        model
            The full MPMModel instance.
        fieldOutputController
            The field output controller.
        outputmanagers
            The list of OutputManagerBase instances.
        userIterationOptions
            The dict controlling the Newton cycle(s).

        Returns
        -------
        tuple[bool, MPMModel]
            The tuple containing:
                - the truth value of success.
                - the updated MPMModel.
        """

        if "arc length parameter" not in model.additionalParameters:
            model.additionalParameters["arc length parameter"] = 0.0
        self._arcLengthController = arcLengthController

        return super().solveStep(
            timeStepper,
            linearSolver,
            mpmManager,
            dirichlets,
            bodyLoads,
            distributedLoads,
            constraints,
            model,
            fieldOutputController,
            outputmanagers,
            userIterationOptions,
        )

    @performancetiming.timeit("newton iteration")
    def _newtonSolve(
        self,
        dirichlets: list,
        bodyLoads: list,
        distributedLoads: list,
        activeNodeSets: list,
        activeCells: list,
        materialPoints: list,
        constraints: list,
        theDofManager: DofManager,
        linearSolver,
        iterationOptions: dict,
        timeStep: TimeStep,
        model: MPMModel,
        newtonCache: tuple = None,
    ) -> tuple[DofVector, DofVector, dict, tuple]:
        """Standard Newton-Raphson scheme to solve for an increment.

        Parameters
        ----------
        dirichlets
            The list of dirichlet StepActions.
        bodyLoads
            The list of bodyload StepActions.
        distributedLoads
            The list of distributed load StepActions.
        activeNodeSets
            The list of (reduced) active NodeSets.
        activeCells
            The list of active cells.
        materialPoints
            The list of material points.
        constraints
            The list of constraints.
        theDofManager
            The DofManager instance for the current active model.
        linearSolver
            The instance of the linear solver to be used.
        iterationOptions
            The specified options controlling the Newton cycle.
        timeStep
            The current time increment.
        model
            The full MPMModel instance.
        newtonCache
            An arbitrary cache of (expensive) objects, which may be reused across time steps as long as the global system does not change.
            If the system changes, the newtonCache is set to None.

        Returns
        -------
        tuple[DofVector,DofVector,dict, object]
            A tuple containing:
                - the new solution increment vector.
                - the current internal load vector.
                - the increment residual history of the Newton cycle.
                - the newton cache which may be resused if the system does not change.
        """

        arcLengthController = self._arcLengthController

        if arcLengthController is None:
            return super()._newtonSolve(
                dirichlets,
                bodyLoads,
                distributedLoads,
                activeNodeSets,
                activeCells,
                materialPoints,
                constraints,
                theDofManager,
                linearSolver,
                iterationOptions,
                timeStep,
                model,
                newtonCache,
            )

        iterationCounter = 0
        incrementResidualHistory = {field: list() for field in theDofManager.idcsOfFieldsInDofVector}

        nAllowedResidualGrowths = iterationOptions["allowed residual growths"]

        if not newtonCache:
            newtonCache = self._createArcLengthNewtonCache(theDofManager)
        K_VIJ, csrGenerator, dU, Rhs_, F, PInt, PExt, PExt_0, PExt_f, K_VIJ_0, K_VIJ_f = newtonCache

        dU[:] = 0.0
        Rhs_0 = Rhs_[:, 0]
        Rhs_f = Rhs_[:, 1]

        ddU = None

        Lambda = model.additionalParameters["arc length parameter"]
        dLambda = 0.0
        ddLambda = 0.0

        referenceTimeStep = TimeStep(timeStep.number, 1.0, 1.0, 0.0, 0.0, 0.0)
        zeroTimeStep = TimeStep(timeStep.number, 0.0, 0.0, 0.0, 0.0, 0.0)

        self._applyStepActionsAtIncrementStart(model, timeStep, dirichlets + bodyLoads)

        while True:
            PInt[:] = K_VIJ[:] = F[:] = PExt_0[:] = PExt_f[:] = K_VIJ_f[:] = K_VIJ_0[:] = 0.0

            self._prepareMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
            self._interpolateFieldsToMaterialPoints(activeCells, dU)
            self._computeMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
            self._computeCells(
                activeCells, dU, PInt, F, K_VIJ, timeStep.totalTime, timeStep.timeIncrement, theDofManager
            )
            self._computeConstraints(constraints, dU, PInt, K_VIJ, timeStep)

            PExt_0, K_VIJ_0 = self._computeBodyLoads(
                bodyLoads, PExt_0, K_VIJ_0, zeroTimeStep, theDofManager, activeCells
            )
            PExt_0, K_VIJ_0 = self._computeDistributedLoads(
                distributedLoads, PExt_0, K_VIJ_0, zeroTimeStep, theDofManager
            )

            PExt_f, K_VIJ_f = self._computeBodyLoads(
                bodyLoads, PExt_f, K_VIJ_f, referenceTimeStep, theDofManager, activeCells
            )
            PExt_f, K_VIJ_f = self._computeDistributedLoads(
                distributedLoads, PExt_f, K_VIJ_f, referenceTimeStep, theDofManager
            )

            PExt_f -= PExt_0  # and subtract the dead part, since we are only interested in the homogeneous linear part
            K_VIJ_f -= K_VIJ_0

            # Dead and Reference load ..
            Rhs_0[:] = -(PExt_0 + (Lambda + dLambda) * PExt_f + PInt)
            Rhs_f[:] = -PExt_f

            # add stiffness contribution
            K_VIJ[:] += K_VIJ_0
            K_VIJ[:] += (Lambda + dLambda) * K_VIJ_f

            Rhs_f = self._applyDirichlet(referenceTimeStep, Rhs_f, dirichlets, activeNodeSets, theDofManager)
            if iterationCounter == 0 and dirichlets:
                Rhs_0 = self._applyDirichlet(timeStep, Rhs_0, dirichlets, activeNodeSets, theDofManager)
            else:
                for dirichlet in dirichlets:
                    Rhs_0[self._findDirichletIndices(theDofManager, dirichlet)] = 0.0

                incrementResidualHistory = self._computeResiduals(
                    Rhs_0, ddU, dU, F, incrementResidualHistory, theDofManager
                )

                converged = self._checkConvergence(iterationCounter, incrementResidualHistory, iterationOptions)

                if converged:
                    break

                if self._checkDivergingSolution(incrementResidualHistory, nAllowedResidualGrowths):
                    self._printResidualOutlierNodes(incrementResidualHistory)
                    raise DivergingSolution("Residual grew {:} times, cutting back".format(nAllowedResidualGrowths))

                if iterationCounter == iterationOptions["max. iterations"]:
                    self._printResidualOutlierNodes(incrementResidualHistory)
                    raise ReachedMaxIterations("Reached max. iterations in current increment, cutting back")

            K_CSR = self._VIJtoCSR(K_VIJ, csrGenerator)
            K_CSR = self._applyDirichletKCsr(K_CSR, dirichlets, theDofManager)

            # solve 2 eq. systems at once:
            ddU_ = self._linearSolve(K_CSR, Rhs_, linearSolver)
            # q_0 = K⁻¹ * -(  Pext_0  + dLambda * Pext_Ref + PInt  )
            # q_f = K⁻¹ * -(  Pext_Ref  )
            ddU_0, ddU_f = ddU_[:, 0], ddU_[:, 1]

            # compute the increment of the load parameter. Method depends on the employed arc length controller
            ddLambda = arcLengthController.computeDDLambda(dU, ddU_0, ddU_f, timeStep, theDofManager)

            # assemble total solution
            ddU = ddU_0 + ddLambda * ddU_f

            dU += ddU
            dLambda += ddLambda

            iterationCounter += 1

        iterationHistory = {"iterations": iterationCounter, "incrementResidualHistory": incrementResidualHistory}

        model.additionalParameters["arc length parameter"] = Lambda + dLambda

        return dU, PInt, iterationHistory, newtonCache

    @performancetiming.timeit("creation newton cache")
    def _createArcLengthNewtonCache(self, theDofManager: DofManager) -> tuple:
        """Create expensive objects, which may be reused if the global system does not change.

        Parameters
        ----------
        theDofManager
            The DofManager instance.

        Returns
        -------
        tuple
            The collection of expensive objects.
        """

        K_VIJ = theDofManager.constructVIJSystemMatrix()
        csrGenerator = self._makeCachedCOOToCSRGenerator(K_VIJ)
        dU = theDofManager.constructDofVector()
        Rhs_ = np.tile(theDofManager.constructDofVector(), (2, 1)).T  # 2 RHSs
        F = theDofManager.constructDofVector()
        PInt = theDofManager.constructDofVector()
        PExt = theDofManager.constructDofVector()
        PExt_0 = theDofManager.constructDofVector()
        PExt_f = theDofManager.constructDofVector()
        K_VIJ_0 = theDofManager.constructVIJSystemMatrix()
        K_VIJ_f = theDofManager.constructVIJSystemMatrix()

        newtonCache = (K_VIJ, csrGenerator, dU, Rhs_, F, PInt, PExt, PExt_0, PExt_f, K_VIJ_0, K_VIJ_f)

        return newtonCache
