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


import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    ConditionalStop,
    DivergingSolution,
    ReachedMaxIncrements,
    ReachedMaxIterations,
    ReachedMinIncrementSize,
    StepFailed,
)
from edelweissfe.utils.fieldoutput import FieldOutputController
from prettytable import PrettyTable

from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.mpmmanagers.base.mpmmanagerbase import MPMManagerBase
from edelweissmpm.particlemanagers.base.baseparticlemanager import BaseParticleManager
from edelweissmpm.solvers.base.nonlinearsolverbase import (
    NonlinearImplicitSolverBase,
    RestartHistoryManager,
)
from edelweissmpm.stepactions.base.mpmbodyloadbase import MPMBodyLoadBase
from edelweissmpm.stepactions.base.mpmdistributedloadbase import MPMDistributedLoadBase
from edelweissmpm.stepactions.particledistributedload import ParticleDistributedLoad


class NonlinearQuasistaticSolver(NonlinearImplicitSolverBase):
    """This is the serial nonlinear implicit quasi static solver.


    Parameters
    ----------
    journal
        The journal instance for logging.
    """

    identification = "MPM-NQS-Solver"

    validOptions = {
        "max. iterations": 10,
        "critical iterations": 5,
        "allowed residual growths": 10,
        "iterations for alt. tolerances": 5,
        "zero increment threshhold": 1e-13,
        "zero flux threshhold": 1e-9,
        "default relative flux residual tolerance": 1e-4,
        "default relative flux residual tolerance alt.": 1e-3,
        "default relative field correction tolerance": 1e-3,
        "default absolute flux residual tolerance": 1e-14,
        "default absolute field correction tolerance": 1e-14,
        "spec. relative flux residual tolerances": dict(),
        "spec. relative flux residual tolerances alt.": dict(),
        "spec. relative field correction tolerances": dict(),
        "spec. absolute flux residual tolerances": dict(),
        "spec. absolute field correction tolerances": dict(),
        "failed increment cutback factor": 0.25,
    }

    def __init__(self, journal: Journal):
        super().__init__(journal)

    @performancetiming.timeit("solve step")
    def solveStep(
        self,
        timeStepper,
        linearSolver,
        model: MPMModel,
        fieldOutputController: FieldOutputController,
        mpmManagers: list[MPMManagerBase] = [],
        particleManagers: list[BaseParticleManager] = [],
        dirichlets: list[DirichletBase] = [],
        bodyLoads: list[MPMBodyLoadBase] = [],
        distributedLoads: list[MPMDistributedLoadBase] = [],
        particleDistributedLoads: list[ParticleDistributedLoad] = [],
        constraints: list[ConstraintBase] = [],
        outputManagers: list[OutputManagerBase] = [],
        userIterationOptions: dict = {},
        vciManagers: list = [],
        restartWriteInterval: int = 0,
        allowFallBackToRestart: bool = False,
        numberOfRestartsToStore=3,
        restartBaseName: str = "restart",
    ) -> tuple[bool, MPMModel]:
        """Public interface to solve for a step.

        Parameters
        ----------
        timeStepper
            The timeStepper instance.
        linearSolver
            The linear solver instance to be used.
        mpmManagers
            The list of MPMManagerBase instances.
        dirichlets
            The list of dirichlet StepActions.
        bodyLoads
            The list of bodyload StepActions.
        distributedLoads
            The list of distributed load StepActions.
        particleDistributedLoads
            The list of particle distributed load StepActions.
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
        vciManagers
            The list of VariationallyConsistentIntegrationManager instances.
        restartWriteInterval
            The interval at which restart files are written.
        allowFallBackToRestart
            The flag to allow a fallback to restart files in case time incrementation fails.
        numberOfRestartsToStore
            The number of restart files to store.
        restartBaseName
            The base name of the restart files.

        Returns
        -------
        tuple[bool, MPMModel]
            The tuple containing:
                - the truth value of success.
                - the updated MPMModel.
        """

        iterationOptions = self.validOptions.copy()

        if userIterationOptions | iterationOptions != iterationOptions:
            raise ValueError("Invalid options in iteration options!")
        iterationOptions.update(userIterationOptions)

        table = PrettyTable(("Solver option", "value"))
        table.add_rows([(k, v) for k, v in iterationOptions.items()])
        table.align = "l"
        self.journal.printPrettyTable(table, self.identification)

        materialPoints = model.materialPoints.values()

        self._applyStepActionsAtStepStart(model, dirichlets + bodyLoads + distributedLoads)

        elements = model.elements.values()
        scalarVariables = model.scalarVariables.values()
        particles = model.particles.values()

        newtonCache = None
        theDofManager = None

        restartHistoryManager = RestartHistoryManager(restartBaseName, numberOfRestartsToStore)

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

                connectivityHasChanged = False

                if materialPoints:
                    self.journal.message(
                        "updating material point - cell connectivity",
                        self.identification,
                        level=1,
                    )
                    self._prepareMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
                    connectivityHasChanged |= self._updateConnectivity(mpmManagers)

                if particleManagers:
                    self.journal.message(
                        "updating particle kernel connectivity",
                        self.identification,
                        level=1,
                    )
                    self._prepareParticles(particles, timeStep.totalTime, timeStep.timeIncrement)
                    connectivityHasChanged |= self._updateConnectivity(particleManagers)

                for c in constraints:
                    connectivityHasChanged |= c.updateConnectivity(model)

                if connectivityHasChanged or not theDofManager:
                    activeCells = set()
                    for man in mpmManagers:
                        activeCells |= man.getActiveCells()

                    self.journal.message(
                        "active domain has changed, (re)initializing equation system & clearing cache",
                        self.identification,
                        level=1,
                    )

                    (activeNodesWithPersistentData, activeNodesWithVolatileData, reducedNodeFields, reducedNodeSets) = (
                        self._assembleActiveDomain(activeCells, model)
                    )

                    theDofManager = self._createDofManager(
                        reducedNodeFields.values(),
                        scalarVariables,
                        elements,
                        constraints,
                        # TODO : Check how to make next line more elegant
                        # TODO 2
                        # list(
                        reducedNodeSets.values(),
                        # ) + [activeNodesWithPersistentData],
                        activeCells,
                        model.cellElements.values(),
                        particles,
                    )

                    U = theDofManager.constructDofVector()

                    # TODO 3
                    # if len(activeNodesWithPersistentData) > 0:
                    #     for field in model.nodeFields.values():
                    #         theDofManager.writeNodeFieldToDofVector(U, field, "U", activeNodesWithPersistentData)

                    self.journal.message(
                        "resulting equation system has a size of {:}".format(theDofManager.nDof),
                        self.identification,
                        level=1,
                    )

                    newtonCache = None

                presentVariableNames = list(theDofManager.idcsOfFieldsInDofVector.keys())

                # if theDofManager.idcsOfScalarVariablesInDofVector:
                #     presentVariableNames += [
                #         "scalar variables",
                #     ]

                # TODO: We are handling this currently as a dedicated special case,
                # but we should consider to make this part of a more general
                # "generic" step action handling once we have more such cases.
                for vciManager in vciManagers:
                    vciManager.computeVCICorrections()

                nVariables = len(presentVariableNames)
                iterationHeader = ("{:^25}" * nVariables).format(*presentVariableNames)
                iterationHeader2 = (" {:<10}  {:<10}  ").format("||R||∞", "||ddU||∞") * nVariables

                self.journal.message(iterationHeader, self.identification, level=2)
                self.journal.message(iterationHeader2, self.identification, level=2)

                try:
                    dU, P, iterationHistory, newtonCache = self._newtonSolve(
                        dirichlets,
                        bodyLoads,
                        distributedLoads,
                        particleDistributedLoads,
                        reducedNodeSets,
                        elements,
                        U,
                        activeCells,
                        model.cellElements.values(),
                        materialPoints,
                        particles,
                        constraints,
                        theDofManager,
                        linearSolver,
                        iterationOptions,
                        timeStep,
                        model,
                        newtonCache,
                    )

                except (RuntimeError, DivergingSolution, ReachedMaxIterations) as e:
                    self.journal.message(str(e), self.identification, 1)

                    try:
                        timeStepper.discardAndChangeIncrement(iterationOptions["failed increment cutback factor"])

                    except ReachedMinIncrementSize:

                        if not allowFallBackToRestart:
                            raise

                        self._tryFallbackWithRestartFiles(restartHistoryManager, timeStepper, model, iterationOptions)

                    for man in outputManagers:
                        man.finalizeFailedIncrement()

                else:
                    if iterationHistory["iterations"] >= iterationOptions["critical iterations"]:
                        timeStepper.preventIncrementIncrease()

                    U += dU

                    # TODO: Make this optional/flexibel via function arguments (?)
                    for field in reducedNodeFields.values():
                        theDofManager.writeDofVectorToNodeField(dU, field, "dU")
                        theDofManager.writeDofVectorToNodeField(U, field, "U")
                        theDofManager.writeDofVectorToNodeField(P, field, "P")
                        model.nodeFields[field.name].copyEntriesFromOther(field)

                    model.advanceToTime(timeStep.totalTime)

                    self.journal.message(
                        "Converged in {:} iteration(s)".format(iterationHistory["iterations"]), self.identification, 1
                    )

                    self._finalizeIncrementOutput(fieldOutputController, outputManagers)

                    if restartWriteInterval and timeStep.number % restartWriteInterval == 0:
                        self.journal.message("Writing restart file", self.identification)

                        restartFileName = restartHistoryManager.getNextRestartFileName()
                        self._writeRestart(model, timeStepper, restartFileName)
                        restartHistoryManager.append(restartFileName)

        except (ReachedMaxIncrements, ReachedMinIncrementSize):
            self.journal.errorMessage("Incrementation failed", self.identification)

        except ConditionalStop:
            self.journal.message("Conditional Stop", self.identification)

        except RuntimeError as e:
            self.journal.errorMessage(str(e), self.identification)
            raise StepFailed()

        self._applyStepActionsAtStepEnd(model, dirichlets + bodyLoads + distributedLoads)

        fieldOutputController.finalizeStep()
        for man in outputManagers:
            man.finalizeStep()

    @performancetiming.timeit("newton iteration")
    def _newtonSolve(
        self,
        dirichlets: list[DirichletBase],
        bodyLoads: list,
        distributedLoads: list,
        particleDistributedLoads: list,
        reducedNodeSets: list,
        elements: list,
        Un: DofVector,
        activeCells: list,
        cellElements: list,
        materialPoints: list,
        particles: list,
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
        elements
            The list of elements.
        Un
            The old solution vector.
        activeCells
            The list of active cells.
        cellElements
            The list of cell elements.
        materialPoints
            The list of material points.
        particles
            The list of particles.
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
        tuple[DofVector,DofVector,dict, tuple]
            A tuple containing:
                - the new solution increment vector.
                - the current internal load vector.
                - the increment residual history of the Newton cycle.
                - a cache of expensive objects which may be reused.
        """

        iterationCounter = 0
        incrementResidualHistory = {field: list() for field in theDofManager.idcsOfFieldsInDofVector}

        nAllowedResidualGrowths = iterationOptions["allowed residual growths"]

        if not newtonCache:
            newtonCache = self._createNewtonCache(theDofManager)
        K_VIJ, csrGenerator, dU, Rhs, F, PInt, PExt = newtonCache

        dU[:] = 0.0
        ddU = None

        self._applyStepActionsAtIncrementStart(model, timeStep, dirichlets + bodyLoads)

        while True:
            PInt[:] = K_VIJ[:] = F[:] = PExt[:] = 0.0

            self._prepareMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
            self._interpolateFieldsToMaterialPoints(activeCells, dU)
            self._interpolateFieldsToMaterialPoints(cellElements, dU)
            self._computeMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)

            self._computeCells(
                activeCells, dU, PInt, F, K_VIJ, timeStep.totalTime, timeStep.timeIncrement, theDofManager
            )

            self._computeElements(
                elements, dU, Un, PInt, F, K_VIJ, timeStep.totalTime, timeStep.timeIncrement, theDofManager
            )

            self._computeCellElements(
                cellElements, dU, Un, PInt, F, K_VIJ, timeStep.totalTime, timeStep.timeIncrement, theDofManager
            )

            self._computeParticles(
                particles, dU, PInt, F, K_VIJ, timeStep.totalTime, timeStep.timeIncrement, theDofManager
            )

            self._computeConstraints(constraints, dU, PInt, K_VIJ, timeStep)

            PExt, K = self._computeBodyLoads(bodyLoads, PExt, K_VIJ, timeStep, theDofManager, activeCells)
            PExt, K = self._computeCellDistributedLoads(distributedLoads, PExt, K_VIJ, timeStep, theDofManager)

            PExt, K = self._computeParticleDistributedLoads(
                particleDistributedLoads, PExt, K_VIJ, timeStep, theDofManager
            )

            Rhs[:] = -PInt
            Rhs -= PExt

            if iterationCounter == 0 and dirichlets:
                Rhs = self._applyDirichlet(timeStep, Rhs, dirichlets, reducedNodeSets, theDofManager)
            else:
                for dirichlet in dirichlets:
                    Rhs[self._findDirichletIndices(theDofManager, dirichlet, reducedNodeSets[dirichlet.nSet])] = 0.0

                incrementResidualHistory = self._computeResiduals(
                    Rhs, ddU, dU, F, incrementResidualHistory, theDofManager
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
            K_CSR = self._applyDirichletKCsr(K_CSR, dirichlets, theDofManager, reducedNodeSets)

            ddU = self._linearSolve(K_CSR, Rhs, linearSolver)
            dU += ddU
            iterationCounter += 1

        iterationHistory = {"iterations": iterationCounter, "incrementResidualHistory": incrementResidualHistory}

        return dU, PInt, iterationHistory, newtonCache
