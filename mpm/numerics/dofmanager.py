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

from fe.numerics.dofmanager import DofManager

from mpm.models.mpmmodel import MPMModel

import numpy as np


class MPMDofManager(DofManager):
    def __init__(
        self, nodeFields: list, scalarVariables: list, elements: list, constraints: list, nodeSets: list, cells: list
    ):
        # self._nNodalFluxesFieldwiseFromCells = self._countNodalFluxesFieldWise(cells)

        # super().__init__(nodeFields, scalarVariables, elements, constraints, nodeSets)

        (
            self.nDof,
            self.idcsOfFieldVariablesInDofVector,
            self.idcsOfFieldsInDofVector,
            self.idcsOfScalarVariablesInDofVector,
        ) = self._initializeDofVectorStructure(nodeFields, scalarVariables)

        self.fields = self.idcsOfFieldsInDofVector.keys()

        self.indexToNodeMapping = self._determineIndexToNodeMap()

        (
            self.accumulatedElementNDof,
            self._accumulatedElementVIJSize,
            self._nAccumulatedNodalFluxesFieldwiseFromElements,
            self.largestNumberOfElNDof,
        ) = self._gatherEntitiesInformation(elements)

        (
            self.accumulatedConstraintNDof,
            self._accumulatedConstraintVIJSize,
            self._nAccumulatedNodalFluxesFieldwiseFromConstraints,
            self.largestNumberOfConstraintNDof,
        ) = self._gatherEntitiesInformation(constraints)

        (
            self.accumulatedCellNDof,
            self.accumulatedCellVIJSize,
            self._nNodalFluxesFieldwiseFromCells,
            self.largestNumberOfCellNDof,
        ) = self._gatherEntitiesInformation(cells)

        self.nAccumulatedNodalFluxesFieldwise = self._computeAccumulatedNodalFluxesFieldWise(self.fields)

        self.idcsOfFieldsOnNodeSetsInDofVector = self._locateFieldsOnNodeSetsInDofVector(nodeSets)
        self.idcsOfElementsInDofVector = self._locateElementsInDofVector(elements)
        self.idcsOfConstraintsInDofVector = self._locateConstraintsInDofVector(constraints)

        self.idcsOfBasicVariablesInDofVector = self._getIndicesOfBasicVariablesInDofVector()

        self.idcsOfCellsInDofVector = self._locateCellsInDofVector(cells)

        self.idcsOfHigherOrderEntitiesInDofVector = self._getIndicesOfAllHigherOrderEntitiesInDofVector()

        self.sizeVIJ = self._computeSizeVIJ()
        (self.I, self.J, self.idcsOfHigherOrderEntitiesInVIJ) = self._initializeVIJPattern()

    def _locateCellsInDofVector(self, cells: list) -> dict:
        """Creates a dictionary containing the location (indices) of each cell
        within the DofVector structure.

        Returns
        -------
        dict
            A dictionary containing the location mapping.
        """

        idcsOfCellsInDofVector = {}

        for cl in cells:
            destList = np.hstack(
                [
                    self.idcsOfFieldVariablesInDofVector[node.fields[nodeField]]
                    for iNode, node in enumerate(cl.nodes)  # for each node of the cell ..
                    for nodeField in cl.fields[iNode]  # for each field of this node
                ]
            )  # the index in the global system

            idcsOfCellsInDofVector[cl] = destList[cl.dofIndicesPermutation]

        return idcsOfCellsInDofVector

    def _computeSizeVIJ(
        self,
    ):
        """Determine the required size of the VIJ system matrix.

        Returns
        -------
        int
            The size of the VIJ system.
        """

        return self.accumulatedCellVIJSize + super()._computeSizeVIJ()

    def _computeAccumulatedNodalFluxesFieldWise(self, fields) -> dict:
        """For the VIJ (COO) system matrix and the Abaqus like convergence test,
        the number of dofs 'entity-wise' is needed:
        = Σ_(elements+constraints) Σ_nodes ( nDof (field) ).

        Returns
        -------
        dict
            Number of accumulated fluxes per field:
                - Field
                - Number of accumulated fluxes
        """
        accumulatedNumberOfFieldFluxes = super()._computeAccumulatedNodalFluxesFieldWise(fields)

        for field in accumulatedNumberOfFieldFluxes.keys():
            accumulatedNumberOfFieldFluxes[field] += self._nNodalFluxesFieldwiseFromCells[field]

        return accumulatedNumberOfFieldFluxes

    def _getIndicesOfAllHigherOrderEntitiesInDofVector(self):
        """
        Get list of indices of all higher order entitties.

        Returns
        -------
        list
            The indices.
        """

        return super()._getIndicesOfAllHigherOrderEntitiesInDofVector() | self.idcsOfCellsInDofVector

    # def _locateCellsInVIJ(
    #     self,
    # ) -> tuple[np.ndarray, np.ndarray, dict]:
    #     """Generate the IJ pattern for VIJ (COO) system matrices.

    #     Returns
    #     -------
    #     tuple
    #          - I vector
    #          - J vector
    #          - the entities to system matrix entry mapping.
    #     """

    #     entitiesInVIJ = {}
    #     entitiesInDofVector = self.idcsOfHigherOrderEntitiesInDofVector

    #     sizeVIJ = self.sizeVIJ

    #     I = np.zeros(sizeVIJ, dtype=int)
    #     J = np.zeros(sizeVIJ, dtype=int)
    #     idxInVIJ = 0

    #     for entity, entityIdcsInDofVector in self.idcsOfHigherOrderEntitiesInDofVector.items():
    #         entitiesInVIJ[entity] = idxInVIJ

    #         nDofEntity = len(entityIdcsInDofVector)

    #         # looks like black magic, but it's an efficient way to generate all indices of Ke in K:
    #         VIJLocations = np.tile(entityIdcsInDofVector, (nDofEntity, 1))
    #         I[idxInVIJ : idxInVIJ + nDofEntity**2] = VIJLocations.flatten()
    #         J[idxInVIJ : idxInVIJ + nDofEntity**2] = VIJLocations.flatten("F")
    #         idxInVIJ += nDofEntity**2

    #     return I, J, entitiesInVIJ
