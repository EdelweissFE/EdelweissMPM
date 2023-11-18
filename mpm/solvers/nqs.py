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
from time import time as getCurrentTime
from collections import defaultdict
from fe.config.linsolve import getLinSolverByName, getDefaultLinSolver
from fe.config.timing import createTimingDict
from fe.config.phenomena import getFieldSize
from fe.numerics.csrgenerator import CSRGenerator
from fe.models.femodel import FEModel
from fe.stepactions.base.stepactionbase import StepActionBase
from fe.timesteppers.timestep import TimeStep
from fe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from fe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from fe.utils.fieldoutput import FieldOutputController
from fe.constraints.base.constraintbase import ConstraintBase
from fe.config.phenomena import fieldCorrectionTolerance, fluxResidualTolerance, fluxResidualToleranceAlternative
from mpm.fields.nodefield import MPMNodeField
from mpm.numerics.dofmanager import MPMDofManager
from mpm.models.mpmmodel import MPMModel
from fe.sets.nodeset import NodeSet
from scipy.sparse import csr_matrix
from numpy import ndarray


class NonlinearQuasistaticSolver:
    """This is the Nonlinear Implicit STatic -- solver.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NonlinearQuasistaticSolver"

    validOptions = {
        "maximumIterations": 10,
        "criticalIterations": 5,
        "allowedResidualGrowths": 10,
        "nMaxIterationsForDefaultTolerances": 5,
    }

    def decorator_timer(name):
        def outer(some_function):
            from time import time

            def inner(self, *args, **kwargs):
                t1 = time()
                result = some_function(self, *args, **kwargs)
                end = time() - t1
                self.computationTimes[name] += end
                return result

            return inner

        return outer

    def __init__(self, journal):
        self.journal = journal
        self.computationTimes = defaultdict(float)

    def solveStep(
        self,
        timeStepper,
        linearSolver,
        mpmManager,
        dirichlets,
        bodyLoads,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: list[OutputManagerBase],
        userIterationOptions: dict,
    ) -> tuple[bool, FEModel]:
        """Public interface to solve for a step.

        Parameters
        ----------
        # stepNumber
        #     The step number.
        step
            The dictionary containing the step definition.
        stepActions
            The dictionary containing all step actions.
        model
            The  model tree.
        fieldOutputController
            The field output controller.
        """

        iterationOptions = self.validOptions.copy()
        iterationOptions["fieldCorrectionTolerances"] = fieldCorrectionTolerance
        iterationOptions["fluxResidualTolerances"] = fluxResidualTolerance
        iterationOptions["fluxResidualTolerancesAlternative"] = fluxResidualToleranceAlternative

        iterationOptions.update(userIterationOptions)

        nMaximumIterations = iterationOptions["nMaximumIterations"]
        nCrititicalIterations = iterationOptions["nCrititcalIterations"]

        materialPoints = model.materialPoints.values()

        self._applyStepActionsAtStepStart(model, dirichlets + bodyLoads)

        try:
            for timeStep in timeStepper.generateTimeStep():
                self.journal.printSeperationLine()
                self.journal.message(
                    "increment {:}: {:8f}, {:8f}; time {:10f} to {:10f}".format(
                        timeStep.number,
                        timeStep.stepProgressIncrement,
                        timeStep.stepProgress,
                        timeStep.totalTime - timeStep.timeIncrement,
                        timeStep.totalTime,
                    ),
                    self.identification,
                    level=1,
                )

                self.journal.message(
                    "updating material point - cell connectivity",
                    self.identification,
                    level=1,
                )

                mpmManager.updateConnectivity()

                if timeStep.number == 0 or mpmManager.hasChanged():
                    self.journal.message(
                        "material points in cells have changed since previous localization",
                        self.identification,
                        level=1,
                    )
                    activeNodes, activeCells, activeNodeFields, activeNodeSets = self._assembleActiveDomain(
                        model, mpmManager
                    )

                    theDofManager = MPMDofManager(
                        activeNodeFields.values(), [], [], [], activeNodeSets.values(), activeCells
                    )

                    presentVariableNames = list(theDofManager.idcsOfFieldsInDofVector.keys())

                    if theDofManager.idcsOfScalarVariablesInDofVector:
                        presentVariableNames += [
                            "scalar variables",
                        ]

                    nVariables = len(presentVariableNames)
                    iterationHeader = ("{:^25}" * nVariables).format(*presentVariableNames)
                    iterationHeader2 = (" {:<10}  {:<10}  ").format("||R||∞", "||ddU||∞") * nVariables

                self.journal.message(iterationHeader, self.identification, level=2)
                self.journal.message(iterationHeader2, self.identification, level=2)

                try:
                    dU, P, iterationHistory = self._solveIncrement(
                        dirichlets,
                        bodyLoads,
                        activeNodeSets,
                        activeCells,
                        materialPoints,
                        theDofManager,
                        linearSolver,
                        iterationOptions,
                        timeStep,
                        model,
                    )

                except CutbackRequest as e:
                    self.journal.message(str(e), self.identification, 1)
                    timeStepper.discardAndChangeIncrement(max(e.cutbackSize, 0.25))

                    for man in outputmanagers:
                        man.finalizeFailedIncrement()

                except (ReachedMaxIterations, DivergingSolution) as e:
                    self.journal.message(str(e), self.identification, 1)
                    timeStepper.discardAndChangeIncrement(0.25)

                    for man in outputmanagers:
                        man.finalizeFailedIncrement()

                else:
                    if iterationHistory["iterations"] >= iterationOptions["nCrititcalIterations"]:
                        timeStepper.preventIncrementIncrease()

                    for field in activeNodeFields.values():
                        theDofManager.writeDofVectorToNodeField(dU, field, "dU")

                    model.nodeFields["displacement"].copyEntriesFromOther(activeNodeFields["displacement"])
                    model.advanceToTime(timeStep.totalTime)

                    self.journal.message(
                        "Converged in {:} iteration(s)".format(iterationHistory["iterations"]), self.identification, 1
                    )

                    fieldOutputController.finalizeIncrement()
                    for man in outputmanagers:
                        man.finalizeIncrement()

        except (ReachedMaxIncrements, ReachedMinIncrementSize):
            self.journal.errorMessage("Incrementation failed", self.identification)
            raise StepFailed()

        except ConditionalStop:
            self.journal.message("Conditional Stop", self.identification)
            self._applyStepActionsAtStepEnd(model, dirichlets + bodyLoads)

        else:
            self._applyStepActionsAtStepEnd(model, dirichlets + bodyLoads)

        finally:
            self.journal.printTable(
                [("Time in {:}".format(k), " {:10.4f}s".format(v)) for k, v in self.computationTimes.items()],
                self.identification,
            )

    @decorator_timer("newton scheme")
    def _solveIncrement(
        self,
        dirichlets,
        bodyLoads,
        activeNodeSets,
        activeCells,
        materialPoints,
        theDofManager,
        linearSolver,
        iterationOptions,
        timeStep: TimeStep,
        model,
    ) -> tuple[DofVector, DofVector, DofVector, int, dict]:
        """Standard Newton-Raphson scheme to solve for an increment.

        Parameters
        ----------
        Un
            The old solution vector.
        dU
            The old solution increment.
        P
            The old reaction vector.
        K
            The system matrix to be used.
        elements
            The dictionary containing all elements.
        stepActions
            The list of active step actions.
        model
            The model tree.
        timeStep.
            The current time increment.

        Returns
        -------
        tuple[DofVector,DofVector,DofVector,int,dict]
            A tuple containing
                - the new solution vector
                - the solution increment
                - the new reaction vector
                - the number of required iterations
                - the history of residuals per field
        """

        iterationCounter = 0
        incrementResidualHistory = {field: list() for field in theDofManager.idcsOfFieldsInDofVector}

        nAllowedResidualGrowths = iterationOptions["nAllowedResidualGrowths"]

        K_VIJ = theDofManager.constructVIJSystemMatrix()
        csrGenerator = CSRGenerator(K_VIJ)

        dU = theDofManager.constructDofVector()
        dU[:] = 0.0
        R = theDofManager.constructDofVector()
        F = theDofManager.constructDofVector()
        P = theDofManager.constructDofVector()
        PExt = theDofManager.constructDofVector()
        ddU = None

        self._applyStepActionsAtIncrementStart(model, timeStep, dirichlets + bodyLoads)

        while True:
            P[:] = K_VIJ[:] = F[:] = PExt[:] = 0.0

            self._prepareMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
            self._interpolateFieldsToMaterialPoints(activeCells, dU)
            self._computeMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
            self._computeCells(activeCells, P, F, K_VIJ, timeStep.totalTime, timeStep.timeIncrement)

            PExt, K = self._computeBodyLoads(bodyLoads, PExt, K_VIJ, timeStep, theDofManager, activeCells)

            R[:] = P
            R += PExt

            if iterationCounter == 0 and dirichlets:
                R = self._applyDirichlet(timeStep, R, dirichlets, activeNodeSets, theDofManager)
            else:
                for dirichlet in dirichlets:
                    R[self._findDirichletIndices(theDofManager, dirichlet)] = 0.0

                incrementResidualHistory = self._computeResiduals(
                    R, ddU, dU, F, incrementResidualHistory, theDofManager
                )

                converged = self._checkConvergence(iterationCounter, incrementResidualHistory, iterationOptions)

                if converged:
                    break

                if self._checkDivergingSolution(incrementResidualHistory, nAllowedResidualGrowths):
                    self._printResidualOutlierNodes(incrementResidualHistory)
                    raise DivergingSolution("Residual grew {:} times, cutting back".format(nAllowedResidualGrowths))

                if iterationCounter == iterationOptions["nMaximumIterations"]:
                    self._printResidualOutlierNodes(incrementResidualHistory)
                    raise ReachedMaxIterations("Reached max. iterations in current increment, cutting back")

            K_CSR = self._VIJtoCSR(K_VIJ, csrGenerator)
            K_CSR = self._applyDirichletKCsr(K_CSR, dirichlets, theDofManager)

            ddU = self._linearSolve(K_CSR, R, linearSolver)
            dU += ddU
            iterationCounter += 1

        iterationHistory = {"iterations": iterationCounter, "incrementResidualHistory": incrementResidualHistory}

        return dU, P, iterationHistory

    @decorator_timer("compute body loads")
    def _computeBodyLoads(
        self,
        bodyForces: list[StepActionBase],
        PExt: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
        theDofManager,
        activeCells,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all body forces loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        U_np
            The current solution vector.
        PExt
            The external load vector to be augmented.
        K
            The system matrix to be augmented.
        timeStep
            The current time increment.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        for bForce in bodyForces:
            loadVector = bForce.getCurrentLoad(timeStep)
            bLoadType = bForce.loadType

            for cl in bForce.cellSet:
                if cl in activeCells:
                    Pc = np.zeros(cl.nDof)
                    Kc = K[theDofManager.idcsOfCellsInDofVector[cl]]

                    cl.computeBodyLoad(bLoadType, loadVector, Pc, Kc, timeStep.totalTime, timeStep.timeIncrement)

                    PExt[cl] += Pc

        return PExt, K

    @decorator_timer("dirichet on CSR")
    def _applyDirichletKCsr(
        self, K: VIJSystemMatrix, dirichlets: list[StepActionBase], theDofManager
    ) -> VIJSystemMatrix:
        """Apply the dirichlet bcs on the global stiffness matrix
        Is called by solveStep() before solving the global system.
        http://stackoverflux.com/questions/12129948/scipy-sparse-set-row-to-zeros

        Parameters
        ----------
        K
            The system matrix.
        dirichlets
            The list of dirichlet boundary conditions.

        Returns
        -------
        VIJSystemMatrix
            The modified system matrix.
        """

        if dirichlets:
            for dirichlet in dirichlets:
                for row in self._findDirichletIndices(theDofManager, dirichlet):  # dirichlet.indices:
                    K.data[K.indptr[row] : K.indptr[row + 1]] = 0.0

            # K[row, row] = 1.0 @ once, faster than within the loop above:
            diag = K.diagonal()
            diag[np.concatenate([self._findDirichletIndices(theDofManager, d) for d in dirichlets])] = 1.0
            K.setdiag(diag)

            K.eliminate_zeros()

        return K

    @decorator_timer("dirichlet on R")
    def _applyDirichlet(
        self, timeStep: TimeStep, R: DofVector, dirichlets: list[StepActionBase], activeNodeSets, theDofManager
    ):
        """Apply the dirichlet bcs on the residual vector
        Is called by solveStep() before solving the global equatuon system.

        Parameters
        ----------
        timeStep
            The time increment.
        R
            The residual vector of the global equation system to be modified.
        dirichlets
            The list of dirichlet boundary conditions.

        Returns
        -------
        DofVector
            The modified residual vector.
        """

        for dirichlet in dirichlets:
            dirichletNodes = activeNodeSets[dirichlet.nSet.name]
            R[self._findDirichletIndices(theDofManager, dirichlet)] = dirichlet.getDelta(
                timeStep, dirichletNodes
            ).flatten()

        return R

    @decorator_timer("evaluation residuals")
    def _computeResiduals(
        self, R: DofVector, ddU: DofVector, dU: DofVector, F: DofVector, residualHistory: dict, theDofManager
    ) -> tuple[bool, dict]:
        """Check the convergence, individually for each field,
        similar to Abaqus based on the current total flux residual and the field correction
        Is called by solveStep() to decide whether to continue iterating or stop.

        Parameters
        ----------
        R
            The current residual.
        ddU
            The current correction increment.
        F
            The accumulated fluxes.
        iterationCounter
            The current iteration number.
        residualHistory
            The previous residuals.

        Returns
        -------
        tuple[bool,dict]
            - True if converged.
            - The residual histories field wise.

        """

        spatialAveragedFluxes = self._computeSpatialAveragedFluxes(F, theDofManager)

        for field, fieldIndices in theDofManager.idcsOfFieldsInDofVector.items():
            fieldResidualAbs = np.abs(R[fieldIndices])

            indexOfMax = np.argmax(fieldResidualAbs)
            fluxResidual = fieldResidualAbs[indexOfMax]
            normalizedFluxResidual = fluxResidual / max(spatialAveragedFluxes[field], 1e-6)
            nodeWithLargestResidual = theDofManager.getNodeForIndexInDofVector(indexOfMax)

            correction = np.linalg.norm(ddU[fieldIndices], np.inf) if ddU is not None else 0.0
            correctionRelative = correction / max(np.linalg.norm(dU[fieldIndices], np.inf), 1e-9)

            residualHistory[field].append(
                {
                    "fluxResidual": fluxResidual,
                    "normalizedFluxResidual": normalizedFluxResidual,
                    "correction": correction,
                    "correctionRelative": correctionRelative,
                    "nodeWithLargestResidual": nodeWithLargestResidual,
                }
            )

        return residualHistory

    def _checkConvergence(self, iterations: int, incrementResidualHistory: dict, iterationOptions: dict):
        iterationMessage = ""
        convergedAtAll = True

        iterationMessageTemplate = "{:11.2e}{:1}{:11.2e}{:1} "

        if iterations < iterationOptions["nMaxIterationsForDefaultTolerances"]:  # standard tolerance set
            fluxResidualTolerances = iterationOptions["fluxResidualTolerances"]
        else:  # alternative tolerance set
            fluxResidualTolerances = iterationOptions["fluxResidualTolerancesAlternative"]

        fieldCorrectionTolerances = iterationOptions["fieldCorrectionTolerances"]

        # for field, fieldIndices in theDofManager.idcsOfFieldsInDofVector.items():
        for field, fieldIncrementResidualHistory in incrementResidualHistory.items():
            correction = fieldIncrementResidualHistory[-1]["correction"]
            correctionRelative = fieldIncrementResidualHistory[-1]["correctionRelative"]

            fluxResidual = fieldIncrementResidualHistory[-1]["fluxResidual"]
            normalizedFluxResidual = fieldIncrementResidualHistory[-1]["normalizedFluxResidual"]

            convergedCorrection = correctionRelative < 1e-3
            convergedNormalizedFlux = normalizedFluxResidual < 1e-3  # fluxResidualTolerances[field]

            iterationMessage += iterationMessageTemplate.format(
                fluxResidual,
                "✓" if convergedNormalizedFlux else " ",
                correction,
                "✓" if convergedCorrection else " ",
            )
            convergedAtAll = convergedAtAll and convergedCorrection and convergedNormalizedFlux

        self.journal.message(iterationMessage, self.identification)

        return convergedAtAll

    @decorator_timer("linear solve")
    def _linearSolve(self, A: csr_matrix, b: DofVector, linearSolver) -> ndarray:
        """Solve the linear equation system.

        Parameters
        ----------
        A
            The system matrix in compressed spare row format.
        b
            The right hand side.

        Returns
        -------
        ndarray
            The solution 'x'.
        """

        ddU = linearSolver(A, b)

        if np.isnan(ddU).any():
            raise DivergingSolution("Obtained NaN in linear solve")

        return ddU

    @decorator_timer("conversion VIJ to CSR")
    def _VIJtoCSR(self, KCoo: VIJSystemMatrix, csrGenerator) -> csr_matrix:
        """Construct a CSR matrix from VIJ format.

        Parameters
        ----------
        K
            The system matrix in VIJ format.
        Returns
        -------
        csr_matrix
            The system matrix in compressed sparse row format.
        """
        KCsr = csrGenerator.updateCSR(KCoo)

        return KCsr

    @decorator_timer("computation spatial fluxes")
    def _computeSpatialAveragedFluxes(self, F: DofVector, theDofManager) -> float:
        """Compute the spatial averaged flux for every field
        Is usually called by checkConvergence().

        Parameters
        ----------
        F
            The accumulated flux vector.

        Returns
        -------
        dict[str,float]
            A dictioary containg the spatial average fluxes for every field.
        """
        spatialAveragedFluxes = dict.fromkeys(theDofManager.idcsOfFieldsInDofVector, 0.0)
        for field, nDof in theDofManager.nAccumulatedNodalFluxesFieldwise.items():
            spatialAveragedFluxes[field] = np.linalg.norm(F[theDofManager.idcsOfFieldsInDofVector[field]], 1) / nDof

        return spatialAveragedFluxes

    @decorator_timer("assembly of loads")
    def _assembleLoads(
        self,
        nodeForces: list[StepActionBase],
        distributedLoads: list[StepActionBase],
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Assemble all loads into a right hand side vector.

        Parameters
        ----------
        nodeForces
            The list of concentrated (nodal) loads.
        distributedLoads
            The list of distributed (surface) loads.
        bodyForces
            The list of body (volumetric) loads.
        U_np
            The current solution vector.
        PExt
            The external load vector.
        K
            The system matrix.
        timeStep
            The current time increment.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            - The modified external load vector.
            - The modified system matrix.
        """
        # cloads
        # for cLoad in nodeForces:
        # # PExt = cLoad.applyOnP(PExt, increment)
        # PExt[self.theDofManager.idcsOfFieldsOnNodeSetsInDofVector[cLoad.field][cLoad.nSet.name]] += cLoad.getLoad(
        #     increment
        # ).flatten()
        # # dloads
        # PExt, K = self.computeDistributedLoads(distributedLoads, U_np, PExt, K, increment)
        PExt, K = self._computeBodyLoads(bodyForces, U_np, PExt, K, timeStep)

        return PExt, K

    # def extrapolateLastIncrement(
    #     self, extrapolation: str, increment: tuple, dU: DofVector, dirichlets: list, lastIncrementSize: float, model
    # ) -> tuple[DofVector, bool]:
    #     """Depending on the current setting, extrapolate the solution of the last increment.

    #     Parameters
    #     ----------
    #     extrapolation
    #         The type of extrapolation.
    #     increment
    #         The current time increment.
    #     dU
    #         The last solution increment.
    #     dirichlets
    #         The list of active dirichlet boundary conditions.
    #     lastIncrementSize
    #         The size of the last increment.

    #     Returns
    #     -------
    #     tuple[DofVector,bool]
    #         - The extrapolated solution increment.
    #         - True if an extrapolation was performed.
    #     """

    #     incNumber, incrementSize, stepProgress, dT, stepTime, totalTime = increment

    #     if extrapolation == "linear" and lastIncrementSize:
    #         dU *= incrementSize / lastIncrementSize
    #         dU = self.applyDirichlet(increment, dU, dirichlets)
    #         isExtrapolatedIncrement = True
    #     else:
    #         isExtrapolatedIncrement = False
    #         dU[:] = 0.0

    #     return dU, isExtrapolatedIncrement

    def _checkDivergingSolution(self, incrementResidualHistory: dict, maxGrowingIter: int) -> bool:
        """Check if the iterative solution scheme is diverging.

        Parameters
        ----------
        incrementResidualHistory
            The dictionary containing the residual history of all fields.
        maxGrowingIter
            The maximum allows number of growths of a residual during the iterative solution scheme.

        Returns
        -------
        bool
            True if solution is diverging.
        """

        for history in incrementResidualHistory.values():
            nGrew = 0
            for i in range(len(history)):
                if history[i]["normalizedFluxResidual"] > history[i - 1]["normalizedFluxResidual"]:
                    nGrew += 1

            if nGrew > maxGrowingIter:
                return True

        return False

    def _printResidualOutlierNodes(self, incrementResidualHistory: dict):
        """Print which nodes have the largest residuals.

        Parameters
        ----------
        residualOutliers
            The dictionary containing the outlier nodes for every field.
        """
        self.journal.message(
            "Residual outliers:",
            self.identification,
            level=1,
        )
        for field, hist in incrementResidualHistory.items():
            self.journal.message(
                "|{:20}|node {:10}|".format(field, hist[-1]["nodeWithLargestResidual"].label),
                self.identification,
                level=2,
            )

    def _applyStepActionsAtStepStart(self, model: FEModel, actions):
        """Called when all step actions should be appliet at the start a step.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The dictionary of active step actions.
        """

        for action in actions:
            action.applyAtStepStart(model)

    def _applyStepActionsAtStepEnd(self, model: FEModel, actions):
        """Called when all step actions should finish a step.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The dictionary of active step actions.
        """

        for action in actions:
            action.applyAtStepEnd(model)

    def _applyStepActionsAtIncrementStart(self, model: FEModel, timeStep: TimeStep, actions):
        """Called when all step actions should be applied at the start of a step.

        Parameters
        ----------
        model
            The model tree.
        increment
            The time increment.
        stepActions
            The dictionary of active step actions.
        """

        for action in actions:
            action.applyAtIncrementStart(model, timeStep)

    def _findDirichletIndices(self, theDofManager, dirichlet):
        fieldIndices = theDofManager.idcsOfFieldsOnNodeSetsInDofVector[dirichlet.field][dirichlet.nSet.name]

        return fieldIndices.reshape((-1, dirichlet.fieldSize))[:, dirichlet.components].flatten()

    @decorator_timer("assembly active domain")
    def _assembleActiveDomain(self, model: MPMModel, mpmManager):
        if mpmManager.hasLostMaterialPoints():
            raise StepFailed("we have lost material points outside the grid!")

        activeCells = mpmManager.getActiveCells()
        for c in activeCells:
            c.assignMaterialPoints(mpmManager.getMaterialPointsInCell(c))

        activeNodes = set([n for cell in activeCells for n in cell.nodes])

        activeNodeFields = {
            nodeField.name: MPMNodeField(nodeField.name, nodeField.dimension, activeNodes)
            for nodeField in model.nodeFields.values()
        }

        activeNodeSets = {
            nodeSet.name: NodeSet(nodeSet.name, activeNodes.intersection(nodeSet))
            for nodeSet in model.nodeSets.values()
        }

        return activeNodes, activeCells, activeNodeFields, activeNodeSets

    @decorator_timer("preparation material points")
    def _prepareMaterialPoints(self, materialPoints: list, time: float, dT: float):
        for mp in materialPoints:
            mp.prepareYourself(time, dT)

    @decorator_timer("interpolation to mps")
    def _interpolateFieldsToMaterialPoints(self, activeCells: list, dU: DofVector):
        for c in activeCells:
            dUCell = dU[c]
            c.interpolateFieldsToMaterialPoints(dUCell)

    @decorator_timer("computation material points")
    def _computeMaterialPoints(self, materialPoints: list, time: float, dT: float):
        for mp in materialPoints:
            mp.computeYourself(time, dT)

    @decorator_timer("computation active cells")
    def _computeCells(
        self, activeCells: list, P: DofVector, F: DofVector, K_VIJ: VIJSystemMatrix, time: float, dT: float
    ):
        for c in activeCells:
            Pc = np.zeros(c.nDof)
            Kc = K_VIJ[c]
            c.computeMaterialPointKernels(Pc, Kc, time, dT)
            P[c] += Pc
            F[c] += abs(Pc)
