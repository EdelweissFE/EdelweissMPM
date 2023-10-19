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
        super().__init__(nodeFields, scalarVariables, elements, constraints, nodeSets)

        self.idcsOfCellsInDofVector = self._locateCellsInDofVector(cells)

        self.idcsOfHigherOrderEntitiesInDofVector |= self.idcsOfCellsInDofVector

        self.idcsInDofVector = self.idcsOfVariablesInDofVector | self.idcsOfHigherOrderEntitiesInDofVector

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
