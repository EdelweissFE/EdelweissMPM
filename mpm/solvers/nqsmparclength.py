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
    CutbackRequest,
    DivergingSolution,
    ConditionalStop,
    StepFailed,
)

# from time import time as getCurrentTime
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

    identification = "MPM-NQS-Solver"

    @performancetiming.timeit("solve step")
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

    #     iterationOptions = self.validOptions.copy()
    #     iterationOptions.update(userIterationOptions)

    #     nMaximumIterations = iterationOptions["max. iterations"]
    #     nCrititicalIterations = iterationOptions["critical iterations"]

    #     materialPoints = model.materialPoints.values()

    #     self._applyStepActionsAtStepStart(model, dirichlets + bodyLoads)

    #     try:
    #         for timeStep in timeStepper.generateTimeStep():
    #             self.journal.printSeperationLine()
    #             self.journal.message(
    #                 "increment {:}: {:8f}, {:8f}; time {:10f} to {:10f}".format(
    #                     timeStep.number,
    #                     timeStep.stepProgressIncrement,
    #                     timeStep.stepProgress,
    #                     timeStep.totalTime - timeStep.timeIncrement,
    #                     timeStep.totalTime,
    #                 ),
    #                 self.identification,
    #                 level=1,
    #             )

    #             self.journal.message(
    #                 "updating material point - cell connectivity",
    #                 self.identification,
    #                 level=1,
    #             )

    #             self._prepareMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
    #             self._updateConnectivity(mpmManager)

    #             for c in constraints:
    #                 c.initializeTimeStep(model, timeStep)

    #             if timeStep.number == 0 or mpmManager.hasChanged():
    #                 self.journal.message(
    #                     "material points in cells have changed since previous localization",
    #                     self.identification,
    #                     level=1,
    #                 )

    #                 activeNodes, activeCells, activeNodeFields, activeNodeSets = self._assembleActiveDomain(
    #                     model, mpmManager
    #                 )

    #                 theDofManager = self._createDofManager(
    #                     activeNodeFields.values(), [], [], constraints, activeNodeSets.values(), activeCells
    #                 )

    #                 presentVariableNames = list(theDofManager.idcsOfFieldsInDofVector.keys())

    #                 # if theDofManager.idcsOfScalarVariablesInDofVector:
    #                 #     presentVariableNames += [
    #                 #         "scalar variables",
    #                 #     ]

    #                 nVariables = len(presentVariableNames)
    #                 iterationHeader = ("{:^25}" * nVariables).format(*presentVariableNames)
    #                 iterationHeader2 = (" {:<10}  {:<10}  ").format("||R||∞", "||ddU||∞") * nVariables

    #             self.journal.message(iterationHeader, self.identification, level=2)
    #             self.journal.message(iterationHeader2, self.identification, level=2)

    #             try:
    #                 dU, P, iterationHistory = self._newtonSolve(
    #                     dirichlets,
    #                     bodyLoads,
    #                     distributedLoads,
    #                     activeNodeSets,
    #                     activeCells,
    #                     materialPoints,
    #                     constraints,
    #                     theDofManager,
    #                     linearSolver,
    #                     iterationOptions,
    #                     timeStep,
    #                     model,
    #                     arcLengthController,
    #                     arcLengthParameter
    #                 )

    #             # except CutbackRequest as e:
    #             #     self.journal.message(str(e), self.identification, 1)
    #             #     timeStepper.discardAndChangeIncrement(max(e.cutbackSize, 0.25))

    #             #     for man in outputmanagers:
    #             #         man.finalizeFailedIncrement()

    #             except Exception as e:
    #                 self.journal.message(str(e), self.identification, 1)
    #                 print(traceback.format_exc())
    #                 timeStepper.discardAndChangeIncrement(0.25)

    #                 for man in outputmanagers:
    #                     man.finalizeFailedIncrement()

    #             else:
    #                 if iterationHistory["iterations"] >= iterationOptions["critical iterations"]:
    #                     timeStepper.preventIncrementIncrease()

    #                 for field in activeNodeFields.values():
    #                     theDofManager.writeDofVectorToNodeField(dU, field, "dU")

    #                 model.nodeFields["displacement"].copyEntriesFromOther(activeNodeFields["displacement"])
    #                 model.advanceToTime(timeStep.totalTime)

    #                 self.journal.message(
    #                     "Converged in {:} iteration(s)".format(iterationHistory["iterations"]), self.identification, 1
    #                 )

    #                 self._finalizeOutput(fieldOutputController, outputmanagers)

    #     except (ReachedMaxIncrements, ReachedMinIncrementSize):
    #         self.journal.errorMessage("Incrementation failed", self.identification)
    #         raise StepFailed()

    #     except ConditionalStop:
    #         self.journal.message("Conditional Stop", self.identification)
    #         self._applyStepActionsAtStepEnd(model, dirichlets + bodyLoads)

    #     else:
    #         self._applyStepActionsAtStepEnd(model, dirichlets + bodyLoads)

    @performancetiming.timeit("solve step", "newton iteration")
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
        # arcLengthController = None
    ) -> tuple[DofVector, DofVector, dict]:
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

        Returns
        -------
        tuple[DofVector,DofVector,dict]
            A tuple containing:
                - the new solution increment vector.
                - the current internal load vector.
                - the increment residual history of the Newton cycle.
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
            )

        iterationCounter = 0
        incrementResidualHistory = {field: list() for field in theDofManager.idcsOfFieldsInDofVector}

        nAllowedResidualGrowths = iterationOptions["allowed residual growths"]

        K_VIJ = theDofManager.constructVIJSystemMatrix()
        csrGenerator = CSRGenerator(K_VIJ)

        dU = theDofManager.constructDofVector()
        dU[:] = 0.0

        R_ = np.tile(theDofManager.constructDofVector(), (2, 1)).T  # 2 RHSs
        R_0 = R_[:, 0]
        R_f = R_[:, 1]
        F = theDofManager.constructDofVector()  # accumulated Flux vector

        P = theDofManager.constructDofVector()
        P_0 = theDofManager.constructDofVector()
        P_f = theDofManager.constructDofVector()
        K_VIJ_f = theDofManager.constructVIJSystemMatrix()
        K_VIJ_0 = theDofManager.constructVIJSystemMatrix()
        ddU = None

        Lambda = model.additionalParameters["arc length parameter"]
        dLambda = 0.0
        ddLambda = 0.0

        referenceTimeStep = TimeStep(timeStep.number, 1.0, 1.0, 0.0, 0.0, 0.0)
        zeroTimeStep = TimeStep(timeStep.number, 0.0, 0.0, 0.0, 0.0, 0.0)

        self._applyStepActionsAtIncrementStart(model, timeStep, dirichlets + bodyLoads)

        while True:
            P[:] = K_VIJ[:] = F[:] = P_0[:] = P_f[:] = K_VIJ_f[:] = K_VIJ_0[:] = 0.0

            self._prepareMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
            self._interpolateFieldsToMaterialPoints(activeCells, dU)
            self._computeMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
            self._computeCells(activeCells, dU, P, F, K_VIJ, timeStep.totalTime, timeStep.timeIncrement, theDofManager)
            self._computeConstraints(constraints, dU, P, K_VIJ, timeStep)

            P_0, K_VIJ_0 = self._computeBodyLoads(bodyLoads, P_0, K_VIJ_0, zeroTimeStep, theDofManager, activeCells)
            P_0, K_VIJ_0 = self._computeDistributedLoads(distributedLoads, P_0, K_VIJ_0, zeroTimeStep, theDofManager)

            P_f, K_VIJ_f = self._computeBodyLoads(
                bodyLoads, P_f, K_VIJ_f, referenceTimeStep, theDofManager, activeCells
            )
            P_f, K_VIJ_f = self._computeDistributedLoads(
                distributedLoads, P_f, K_VIJ_f, referenceTimeStep, theDofManager
            )

            P_f -= P_0  # and subtract the dead part, since we are only interested in the homogeneous linear part
            K_VIJ_f -= K_VIJ_0

            # Dead and Reference load ..
            R_0[:] = P_0 + (Lambda + dLambda) * P_f + P
            R_f[:] = P_f

            # add stiffness contribution
            K_VIJ[:] += K_VIJ_0
            K_VIJ[:] += (Lambda + dLambda) * K_VIJ_f

            R_f = self._applyDirichlet(referenceTimeStep, R_f, dirichlets, activeNodeSets, theDofManager)
            if iterationCounter == 0 and dirichlets:
                R_0 = self._applyDirichlet(timeStep, R_0, dirichlets, activeNodeSets, theDofManager)
            else:
                for dirichlet in dirichlets:
                    R_0[self._findDirichletIndices(theDofManager, dirichlet)] = 0.0

                incrementResidualHistory = self._computeResiduals(
                    R_0, ddU, dU, F, incrementResidualHistory, theDofManager
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
            ddU_ = self._linearSolve(K_CSR, -R_, linearSolver)
            # q_0 = K⁻¹ * (  Pext_0  + dLambda * Pext_Ref - PInt  )
            # q_f = K⁻¹ * (  Pext_Ref  )
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

        return dU, P, iterationHistory
