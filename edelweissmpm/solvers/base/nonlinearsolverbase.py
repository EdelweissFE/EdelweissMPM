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

from abc import abstractmethod
from collections import deque

import edelweissfe.utils.performancetiming as performancetiming
import h5py
import numpy as np
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.csrgenerator import CSRGenerator
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    DivergingSolution,
    ReachedMinIncrementSize,
    StepFailed,
)
from edelweissfe.utils.fieldoutput import FieldOutputController
from numpy import ndarray
from scipy.sparse import csr_matrix

from edelweissmpm.fields.nodefield import MPMNodeField
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.mpmmanagers.base.mpmmanagerbase import MPMManagerBase
from edelweissmpm.numerics.dofmanager import MPMDofManager
from edelweissmpm.particlemanagers.base.baseparticlemanager import BaseParticleManager
from edelweissmpm.stepactions.base.mpmbodyloadbase import MPMBodyLoadBase
from edelweissmpm.stepactions.base.mpmdistributedloadbase import MPMDistributedLoadBase
from edelweissmpm.stepactions.particledistributedload import ParticleDistributedLoad


class NonlinearImplicitSolverBase:
    """This is the base class for nonlinear implicit solvers.


    Parameters
    ----------
    journal
        The journal instance for logging.
    """

    def __init__(self, journal: Journal):
        self.journal = journal

    @abstractmethod
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
    ) -> tuple[bool, MPMModel]:
        pass

    @abstractmethod
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
        pass

    @performancetiming.timeit("compute body loads")
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

    @performancetiming.timeit("compute distributed loads")
    def _computeCellDistributedLoads(
        self,
        distributedLoads: list[MPMDistributedLoadBase],
        PExt: DofVector,
        K_VIJ: VIJSystemMatrix,
        timeStep: TimeStep,
        theDofManager,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all body forces loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        PExt
            The external load vector to be augmented.
        K_VIJ
            The system matrix to be augmented.
        timeStep
            The current time increment.
        theDofManager
            The DofManager instance.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        for distributedLoad in distributedLoads:
            for mp in distributedLoad.mpSet:
                surfaceID, loadVector = distributedLoad.getCurrentMaterialPointLoad(mp, timeStep)

                for cl in mp.assignedCells:
                    Pc = np.zeros(cl.nDof)
                    Kc = K_VIJ[cl]
                    cl.computeDistributedLoad(
                        distributedLoad.loadType,
                        surfaceID,
                        mp,
                        loadVector,
                        Pc,
                        Kc,
                        timeStep.totalTime,
                        timeStep.timeIncrement,
                    )
                    PExt[cl] += Pc

        return PExt, K_VIJ

    @performancetiming.timeit("compute distributed loads")
    def _computeParticleDistributedLoads(
        self,
        distributedLoads: list[ParticleDistributedLoad],
        PExt: DofVector,
        K_VIJ: VIJSystemMatrix,
        timeStep: TimeStep,
        theDofManager,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all body forces loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        PExt
            The external load vector to be augmented.
        K_VIJ
            The system matrix to be augmented.
        timeStep
            The current time increment.
        theDofManager
            The DofManager instance.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        for distributedLoad in distributedLoads:
            surfaceID, loadVector = distributedLoad.getCurrentLoad(None, timeStep)
            for p in distributedLoad.particles:
                Pc = np.zeros(p.nDof)
                Kc = K_VIJ[p]
                Kc2 = np.zeros_like(Kc)
                p.computeDistributedLoad(
                    distributedLoad.loadType,
                    surfaceID,
                    loadVector,
                    Pc,
                    Kc,
                    timeStep.totalTime,
                    timeStep.timeIncrement,
                )
                p.computeDistributedLoad(
                    distributedLoad.loadType,
                    surfaceID,
                    loadVector,
                    Pc,
                    Kc2,
                    timeStep.totalTime,
                    timeStep.timeIncrement,
                )

                PExt[p] += Pc

        return PExt, K_VIJ

    @performancetiming.timeit("dirichlet on CSR")
    def _applyDirichletKCsr(
        self,
        K: VIJSystemMatrix,
        dirichlets: list[DirichletBase],
        theDofManager: DofManager,
        reducedNodeSets: dict[NodeSet, NodeSet],
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
                reducedNodeSet = reducedNodeSets[dirichlet.nSet]
                for row in self._findDirichletIndices(theDofManager, dirichlet, reducedNodeSet):  # dirichlet.indices:
                    K.data[K.indptr[row] : K.indptr[row + 1]] = 0.0

            # K[row, row] = 1.0 @ once, faster than within the loop above:
            diag = K.diagonal()
            diag[
                np.concatenate(
                    [self._findDirichletIndices(theDofManager, d, reducedNodeSets[d.nSet]) for d in dirichlets]
                )
            ] = 1.0
            K.setdiag(diag)

            K.eliminate_zeros()

        return K

    @performancetiming.timeit("dirichlet on R")
    def _applyDirichlet(
        self,
        timeStep: TimeStep,
        R: DofVector,
        dirichlets: list[DirichletBase],
        reducedNodeSets,
        theDofManager: DofManager,
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
        activeNodeSets
            The sets with active nodes only.
        theDofManager
            The DofManager instance.

        Returns
        -------
        DofVector
            The modified residual vector.
        """

        for dirichlet in dirichlets:
            dirichletNodes = reducedNodeSets[dirichlet.nSet]
            R[self._findDirichletIndices(theDofManager, dirichlet, dirichletNodes)] = dirichlet.getDelta(
                timeStep, dirichletNodes
            ).flatten()

        return R

    @performancetiming.timeit("evaluation residuals")
    def _computeResiduals(
        self,
        R: DofVector,
        ddU: DofVector,
        dU: DofVector,
        F: DofVector,
        residualHistory: dict,
        theDofManager: DofManager,
    ) -> tuple[bool, dict]:
        """Compute the current residuals and relative flux residuals flux (R) and effort correction (ddU).

        Parameters
        ----------
        R
            The current residual.
        ddU
            The current correction increment.
        dU
            The current solution increment.
        F
            The accumulated fluxes.
        residualHistory
            The previous residuals.
        theDofManager
            The DofManager instance.

        Returns
        -------
        tuple[bool,dict]
            - True if converged.
            - The residual histories field wise.

        """

        if np.isnan(R).any():
            raise DivergingSolution("NaN obtained in residual.")

        spatialAveragedFluxes = self._computeSpatialAveragedFluxes(F, theDofManager)

        for field, fieldIndices in theDofManager.idcsOfFieldsInDofVector.items():
            fieldResidualAbs = np.abs(R[fieldIndices])

            indexOfMax = np.argmax(fieldResidualAbs)
            fluxResidualAbsolute = fieldResidualAbs[indexOfMax]

            fluxResidualRelative = fluxResidualAbsolute / max(spatialAveragedFluxes[field], 1e-16)
            nodeWithLargestResidual = theDofManager.getNodeForIndexInDofVector(indexOfMax)

            maxIncrement = np.linalg.norm(dU[fieldIndices], np.inf)
            correctionAbsolute = np.linalg.norm(ddU[fieldIndices], np.inf) if ddU is not None else 0.0
            correctionRelative = correctionAbsolute / max(maxIncrement, 1e-16)

            residualHistory[field].append(
                {
                    "absolute flux residual": fluxResidualAbsolute,
                    "spatial average flux": spatialAveragedFluxes[field],
                    "relative flux residual": fluxResidualRelative,
                    "max. increment": maxIncrement,
                    "absolute correction": correctionAbsolute,
                    "relative correction": correctionRelative,
                    "node with largest residual": nodeWithLargestResidual,
                }
            )

        return residualHistory

    @performancetiming.timeit("convergence check")
    def _checkConvergence(self, iterations: int, incrementResidualHistory: dict, iterationOptions: dict) -> bool:
        """Check the status of convergence.

        Parameters
        ----------
        iterations
            The current number of iterations.
        incrementResidualHistory
            The dictionary containing information about all residuals (and history) for the current Newton cycle.
        iterationOptions
            The dictionary containing settings controlling the convergence tolerances.

        Returns
        -------
        bool
            The truth value of convergence."""

        iterationMessage = ""
        convergedAtAll = True

        iterationMessageTemplate = "{:11.2e}{:1}{:11.2e}{:1} "

        useStrictFluxTolerances = iterations < iterationOptions["iterations for alt. tolerances"]

        for field, fieldIncrementResidualHistory in incrementResidualHistory.items():
            lastResults = fieldIncrementResidualHistory[-1]
            correctionAbs = lastResults["absolute correction"]
            correctionRel = lastResults["relative correction"]

            fluxResidualAbs = lastResults["absolute flux residual"]
            fluxResidualRel = lastResults["relative flux residual"]
            spatialAveragedFlux = lastResults["spatial average flux"]

            if useStrictFluxTolerances:
                fluxTolRel = iterationOptions["spec. relative flux residual tolerances"].get(
                    field, iterationOptions["default relative flux residual tolerance"]
                )
            else:
                fluxTolRel = iterationOptions["spec. relative flux residual tolerances alt."].get(
                    field, iterationOptions["default relative flux residual tolerance alt."]
                )

            fluxTolAbs = iterationOptions["spec. absolute flux residual tolerances"].get(
                field, iterationOptions["default absolute flux residual tolerance"]
            )

            correctionTolRel = iterationOptions["spec. relative field correction tolerances"].get(
                field, iterationOptions["default relative field correction tolerance"]
            )

            correctionTolAbs = iterationOptions["spec. absolute field correction tolerances"].get(
                field, iterationOptions["default absolute field correction tolerance"]
            )

            nonZeroIncrement = lastResults["max. increment"] > iterationOptions["zero increment threshhold"]
            convergedCorrection = correctionRel < correctionTolRel if nonZeroIncrement else True
            convergedCorrection = convergedCorrection or correctionAbs < correctionTolAbs

            nonZeroFlux = spatialAveragedFlux > iterationOptions["zero flux threshhold"]
            convergedFlux = fluxResidualRel < fluxTolRel if nonZeroFlux else True
            convergedFlux = convergedFlux or fluxResidualAbs < fluxTolAbs

            if iterations == 0:
                convergedFlux = False

            iterationMessage += iterationMessageTemplate.format(
                fluxResidualAbs,
                "✓" if convergedFlux else " ",
                correctionAbs,
                "✓" if convergedCorrection else " ",
            )
            convergedAtAll = convergedAtAll and convergedCorrection and convergedFlux

        self.journal.message(iterationMessage, self.identification)

        return convergedAtAll

    @performancetiming.timeit("linear solve")
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

    @performancetiming.timeit("conversion VIJ to CSR")
    def _VIJtoCSR(self, KCoo: VIJSystemMatrix, csrGenerator) -> csr_matrix:
        """Construct a CSR matrix from VIJ (COO)format.

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

    @performancetiming.timeit("computation spatial fluxes")
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
                if history[i]["relative flux residual"] > history[i - 1]["relative flux residual"]:
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
                "|{:20}|node {:10}|".format(field, hist[-1]["node with largest residual"].label),
                self.identification,
                level=2,
            )

    @performancetiming.timeit("step actions")
    def _applyStepActionsAtStepStart(self, model: MPMModel, actions):
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

    @performancetiming.timeit("step actions")
    def _applyStepActionsAtStepEnd(self, model: MPMModel, actions):
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

    @performancetiming.timeit("step actions")
    def _applyStepActionsAtIncrementStart(self, model: MPMModel, timeStep: TimeStep, actions):
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

    def _findDirichletIndices(self, theDofManager, dirichlet, reducedNodeSet):
        fieldIndices = theDofManager.idcsOfFieldsOnNodeSetsInDofVector[dirichlet.field][reducedNodeSet]

        return fieldIndices.reshape((-1, dirichlet.fieldSize))[:, dirichlet.components].flatten()

    @performancetiming.timeit("assembly active domain")
    def _assembleActiveDomain(self, activeCells, model: MPMModel) -> tuple[NodeSet, NodeSet, list, list]:
        """Gather the Nodes, active NodeFields and NodeSets.

        Parameters
        ----------
        model
            The full MPMModel.
        mpmManager
            The MPMManager intance.

        Returns
        -------
        tuple
            The tuple containing:
                - The set of active Nodes with persistent field values (FEM).
                - The set of active Nodes with volatile field values (MPM).
                - the list of NodeFields on the active Nodes.
                - the list of reduced NodeSets on the active Nodes.
        """
        # TODO: This method should be part of Model, in the spirit of 'getReducedModel()' or similar

        activeNodesWithPersistentFieldValues = set(
            n for element in model.elements.values() for n in element.nodes
        ) | set(n for element in model.cellElements.values() for n in element.nodes)

        activeNodesWithVolatileFieldValues = set(n for cell in activeCells for n in cell.nodes)

        activeNodesWithVolatileFieldValues |= set(
            kf.node for particle in model.particles.values() for kf in particle.kernelFunctions
        )

        activeNodes = activeNodesWithVolatileFieldValues | activeNodesWithPersistentFieldValues

        activeNodes = NodeSet("activeNodes", activeNodes)
        activeNodesWithPersistentFieldValues = NodeSet(
            "activeNodesWithPersistentFieldvalues", activeNodesWithPersistentFieldValues
        )
        activeNodesWithVolatileFieldValues = NodeSet(
            "activeNodesWithVolatileFieldValues", activeNodesWithVolatileFieldValues
        )

        reducedNodeFields = {
            nodeField.name: MPMNodeField(nodeField.name, nodeField.dimension, activeNodes)
            for nodeField in model.nodeFields.values()
        }

        reducedNodeSets = {
            nodeSet: NodeSet(nodeSet.name, set(activeNodes).intersection(nodeSet))
            for nodeSet in model.nodeSets.values()
        }

        return (
            activeNodesWithPersistentFieldValues,
            activeNodesWithVolatileFieldValues,
            reducedNodeFields,
            reducedNodeSets,
        )

    @performancetiming.timeit("preparation material points")
    def _prepareMaterialPoints(self, materialPoints: list, time: float, dT: float):
        """Let the material points know that a new time step begins.

        Parameters
        ----------
        materialPoints
            The list of material points to be prepared.
        time
            The current time.
        dT
            The current time increment.
        """
        for mp in materialPoints:
            mp.prepareYourself(time, dT)

    @performancetiming.timeit("interpolation to mps")
    def _interpolateFieldsToMaterialPoints(self, activeCells: list, dU: DofVector):
        """Let the solution be interpolated to all material points using the cells.

        Parameters
        ----------
        activeCells
            The list of active cells, which contain material points.
        dU
            The current solution increment to be interpolated.
        """
        for c in activeCells:
            dUCell = dU[c]
            c.interpolateFieldsToMaterialPoints(dUCell)

    @performancetiming.timeit("computation material points")
    def _computeMaterialPoints(self, materialPoints: list, time: float, dT: float):
        """Evaluate all material points' physics.

        Parameters
        ----------
        materialPonts
            The list material points to  evaluated.
        time
            The current time.
        dT
            The increment of time.
        """
        for mp in materialPoints:
            mp.computeYourself(time, dT)

    @performancetiming.timeit("computation active cells")
    def _computeCells(
        self,
        activeCells: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        activeCells
            The list of (active) cells to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        for c in activeCells:
            dUc = dU[c]
            Pc = np.zeros(c.nDof)
            Kc = K_VIJ[c]
            c.computeMaterialPointKernels(dUc, Pc, Kc, time, dT)
            P[c] += Pc
            F[c] += abs(Pc)

    @performancetiming.timeit("computation active cells")
    def _computeCellsWithLumpedInertia(
        self,
        activeCells: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        M: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        activeCells
            The list of (active) cells to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        for c in activeCells:
            dUc = dU[c]
            Pc = np.zeros(c.nDof)
            Mc = np.zeros(c.nDof)
            Kc = K_VIJ[c]
            c.computeMaterialPointKernels(dUc, Pc, Kc, time, dT)
            c.computeLumpedInertia(Mc)
            M[c] += Mc
            P[c] += Pc
            F[c] += abs(Pc)

    @performancetiming.timeit("computation active cells")
    def _computeCellsWithConsistentInertia(
        self,
        activeCells: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        M: VIJSystemMatrix,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        activeCells
            The list of (active) cells to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        for c in activeCells:
            dUc = dU[c]
            Pc = np.zeros(c.nDof)
            Mc = np.zeros((c.nDof, c.nDof))
            Kc = K_VIJ[c]
            c.computeMaterialPointKernels(dUc, Pc, Kc, time, dT)
            c.computeConsistentInertia(Mc)
            M[c] += Mc
            P[c] += Pc
            F[c] += abs(Pc)

    @performancetiming.timeit("computation elements")
    def _computeElements(
        self,
        elements: list,
        dU: DofVector,
        Un: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        Un
            The previous global solution vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        time_ = np.array([time, time])

        for el in elements:
            dUEl = dU[el]
            UEln = Un[el]
            UElnp = UEln + dUEl
            PEl = np.zeros(el.nDof)
            KEl = K_VIJ[el]
            el.computeYourself(KEl, PEl, UElnp, dUEl, time_, dT)
            P[el] -= PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation elements")
    def _computeElementsWithConsistentInertia(
        self,
        elements: list,
        dU: DofVector,
        Un: DofVector,
        P: DofVector,
        F: DofVector,
        M_VIJ: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all elements with consistent inertia.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        Un
            The previous global solution vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        M_VIJ
            The global consistent mass matrix in VIJ (COO) format.
        K_VIJ
            The global system stiffness matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        time_ = np.array([time, time])

        for el in elements:
            dUEl = dU[el]
            UEln = Un[el]
            UElnp = UEln + dUEl
            PEl = np.zeros(el.nDof)
            MEl = np.zeros_like(M_VIJ[el])
            KEl = K_VIJ[el]
            el.computeYourself(KEl, PEl, UElnp, dUEl, time_, dT)
            el.computeConsistentInertia(MEl)
            M_VIJ[el] += MEl
            P[el] -= PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation elements")
    def _computeElementsWithLumpedInertia(
        self,
        elements: list,
        dU: DofVector,
        Un: DofVector,
        P: DofVector,
        F: DofVector,
        M: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all elements with lumped inertia.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        Un
            The previous global solution vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        M
            The lumped mass matrix as vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        time_ = np.array([time, time])

        for el in elements:
            dUEl = dU[el]
            UEln = Un[el]
            UElnp = UEln + dUEl
            PEl = np.zeros(el.nDof)
            MEl = np.zeros(el.nDof)
            KEl = K_VIJ[el]
            el.computeYourself(KEl, PEl, UElnp, dUEl, time_, dT)
            el.computeLumpedInertia(MEl)
            M[el] += MEl
            P[el] -= PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation cell elements")
    def _computeCellElements(
        self,
        elements: list,
        dU: DofVector,
        Un: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        Un
            The previous global solution vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """

        for el in elements:
            dUEl = dU[el]
            UEln = Un[el]
            UElnp = UEln + dUEl
            PEl = np.zeros(el.nDof)
            KEl = K_VIJ[el]
            el.computeMaterialPointKernels(UElnp, PEl, KEl, time, dT)
            P[el] += PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation particles")
    def _computeParticles(
        self,
        particles: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        for p in particles:
            dUP = dU[p]
            PP = np.zeros(p.nDof)
            KP = K_VIJ[p]
            p.computePhysicsKernels(dUP, PP, KP, time, dT)
            P[p] += PP
            F[p] += abs(PP)

    @performancetiming.timeit("computation particles")
    def _computeParticlesWithConsistentInertia(
        self,
        particles: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        M_VIJ: VIJSystemMatrix,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all particles with consistent inertia.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        M_VIJ
            The global mass matrix in VIJ (COO) format.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        for p in particles:
            dUP = dU[p]
            PP = np.zeros(p.nDof)
            MP = np.zeros_like(M_VIJ[p])
            KP = K_VIJ[p]
            p.computePhysicsKernels(dUP, PP, KP, time, dT)
            p.computeConsistentInertia(MP)
            MP_ = M_VIJ[p]
            MP_ += MP
            P[p] += PP
            F[p] += abs(PP)

    @performancetiming.timeit("computation particles")
    def _computeParticlesWithLumpedInertia(
        self,
        particles: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        M: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all particles with consistent inertia.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        M
            The global lumped mass matrix as vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        for p in particles:
            dUP = dU[p]
            PP = np.zeros(p.nDof)
            MP = np.zeros(p.nDof)
            KP = K_VIJ[p]
            p.computePhysicsKernels(dUP, PP, KP, time, dT)
            p.computeLumpedInertia(MP)
            M[p] += MP
            P[p] += PP
            F[p] += abs(PP)

    @performancetiming.timeit("computation constraints")
    def _computeConstraints(
        self, constraints: list, dU: DofVector, P: DofVector, K_VIJ: VIJSystemMatrix, timeStep: TimeStep
    ):
        """Evaluate all constraints.

        Parameters
        ----------
        constraints
            The list of constraints to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        timeStep
            The current time increment.
        """
        for c in constraints:
            dUc = dU[c]
            Pc = np.zeros(c.nDof)
            Kc = K_VIJ[c]
            c.applyConstraint(dUc, Pc, Kc, timeStep)
            P[c] += Pc

    @performancetiming.timeit("instancing dof manager")
    def _createDofManager(self, *args):
        return MPMDofManager(*args)

    @performancetiming.timeit("update connectivity")
    def _updateConnectivity(self, managers: list[MPMManagerBase] | list[BaseParticleManager]) -> bool:
        """Update the connectivity of all MPMManagers or particle managers.

        Parameters
        ----------
        managers
            The list of managers to update.
        """
        connectivityHasChanged = False
        for man in managers:
            connectivityHasChanged |= man.updateConnectivity()

        return connectivityHasChanged

    @performancetiming.timeit("instancing csr generator")
    def _makeCachedCOOToCSRGenerator(self, K_VIJ):
        return CSRGenerator(K_VIJ)

    @performancetiming.timeit("postprocessing & output")
    def _finalizeIncrementOutput(self, fieldOutputController, outputmanagers):
        fieldOutputController.finalizeIncrement()
        for man in outputmanagers:
            man.finalizeIncrement()

    @performancetiming.timeit("creation newton cache")
    def _createNewtonCache(self, theDofManager):
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
        Rhs = theDofManager.constructDofVector()
        F = theDofManager.constructDofVector()
        PInt = theDofManager.constructDofVector()
        PExt = theDofManager.constructDofVector()

        newtonCache = (K_VIJ, csrGenerator, dU, Rhs, F, PInt, PExt)

        return newtonCache

    @performancetiming.timeit("writing restart")
    def _writeRestart(self, model: MPMModel, timeStepper, fileName):
        """Write the restart file.

        Parameters
        ----------
        model
            The model to be written.
        timeStepper
            The timeStepper to be written.
        fileName
            The name of the restart file.
        """
        theRestartFile = h5py.File(fileName, "w")

        model.writeRestart(theRestartFile)
        timeStepper.writeRestart(theRestartFile)

    def readRestart(
        self,
        restartFile,
        timeStepper,
        model: MPMModel,
    ):
        """Read a restart file.

        Parameters
        ----------
        restartFile
            The name of the restart file.
        timeStepper
            The timeStepper instance to be read from the restart file.
        model
            The full MPMModel instance to be read from the restart file.
        """
        theRestartFile = h5py.File(restartFile, "r")

        model.readRestart(theRestartFile)
        timeStepper.readRestart(theRestartFile)

    def _tryFallbackWithRestartFiles(self, writtenRestarts, timeStepper, model, iterationOptions):
        """Fallback to a previous converged increment using the written restart files.

        Parameters
        ----------
        writtenRestarts
            The list of written restart files.
        timeStepper
            The timeStepper instance.
        model
            The full MPMModel instance.
        iterationOptions
            The dictionary containing the iteration options.
        """

        while True:
            try:
                previousRestartFile = writtenRestarts.pop()
            except IndexError:
                raise StepFailed("No more restart files available for fallback")

            self.readRestart(previousRestartFile, timeStepper, model)
            self.journal.message(
                "Reverting to last successful increment at time {:}".format(model.time), self.identification
            )

            try:
                timeStepper.reduceNextIncrement(iterationOptions["failed increment cutback factor"])
            except ReachedMinIncrementSize:
                continue

            break


class RestartHistoryManager(deque):

    def __init__(self, restartBaseName, maxsize):
        super().__init__(maxlen=maxsize)
        self._restartBaseName = restartBaseName
        self._maxsize = maxsize
        self._currentCount = 0

    def append(self, item):
        super().append(item)
        self._currentCount = (self._currentCount + 1) % self._maxsize

    def pop(self):
        self._currentCount = self._currentCount - 1 if self._currentCount > 0 else self._maxsize - 1
        return super().pop()

    def getNextRestartFileName(
        self,
    ):
        theFileName = "{:}_{:}.h5".format(self._restartBaseName, self._currentCount)
        return theFileName
