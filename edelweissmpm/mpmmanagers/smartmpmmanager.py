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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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

from collections import defaultdict

import numpy as np

from edelweissmpm.cells.base.cell import CellBase
from edelweissmpm.materialpoints.base.mp import MaterialPointBase
from edelweissmpm.mpmmanagers.base.mpmmanagerbase import MPMManagerBase
from edelweissmpm.mpmmanagers.utils import BoundingBox, KDTree, buildBoundingBoxFromCells


class SmartMaterialPointManager(MPMManagerBase):
    """A smart manager for material points and cells making use of a KDTree for loacation points in cells.

    Parameters
    ----------
    materialPointCells
        The list of cells.
    materialPoints
        The list of material points.
    dimension
        The dimension of the problem.
    options
        A dictionary containing options.
    """

    def __init__(
        self,
        materialPointCells: list[CellBase],
        materialPoints: list[MaterialPointBase],
        dimension: int,
        options: dict = {"KDTreeLevels": 1},
    ):
        self._cells = materialPointCells
        self._mps = materialPoints
        self._options = options
        self._activeCells = defaultdict(list)

        self._KDTree = KDTree(
            buildBoundingBoxFromCells(materialPointCells, dimension),
            self._options.get("KDTreeLevels"),
            materialPointCells,
        )

    def updateConnectivity(
        self,
    ):
        activeCells = defaultdict(list)

        hasChanged = False

        for mp in self._mps:
            mpCells = [
                self._KDTree.getCellForCoordinates(vertexCoord, initialGuess=mp.assignedCells)
                for vertexCoord in mp.getVertexCoordinates()
            ]

            mp.assignCells(mpCells)

            for cell in mp.assignedCells:
                activeCells[cell].append(mp)

        if activeCells.keys() != self._activeCells.keys():
            hasChanged = True

        self._activeCells = activeCells
        for c, mps in self._activeCells.items():
            c.assignMaterialPoints(mps)

        return hasChanged

    def getActiveCells(
        self,
    ):
        return self._activeCells.keys()
