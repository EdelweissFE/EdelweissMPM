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

from mpm.cells.base.cell import CellBase
from mpm.materialpoints.base.mp import MaterialPointBase
from mpm.mpmmanagers.utils import KDTree, BoundingBox, buildBoundingBoxFromCells
from mpm.mpmmanagers.base.mpmmanagerbase import MPMManagerBase

from collections import defaultdict

import numpy as np


class SmartMaterialPointManager(MPMManagerBase):
    """A smart manager for material points and cells making use of a KDTree for loacation points in cells.

    Parameters
    ----------
    materialPointCells
        The list of CellBases
    materialPoints
        The list of Materialpoints
    options
        A dictionary containing options
    """

    def __init__(
        self,
        materialPointCells: list[CellBase],
        materialPoints: list[MaterialPointBase],
        options: dict = {"KDTreeLevels": 1},
    ):
        self._cells = materialPointCells
        self._mps = materialPoints
        self._options = options
        self._activeCells = defaultdict(list)

        self._KDTree = KDTree(
            buildBoundingBoxFromCells(materialPointCells),
            self._options.get("KDTreeLevels"),
            materialPointCells,
        )

    def updateConnectivity(
        self,
    ):
        self._activeCells = defaultdict(list)

        for mp in self._mps:
            mp.assignCells(
                [self._KDTree.getCellForCoordinates(vertexCoord) for vertexCoord in mp.getVertexCoordinates()]
            )
            for cell in mp.assignedCells:
                self._activeCells[cell].append(mp)

        for c, mps in self._activeCells.items():
            c.assignMaterialPoints(mps)

    def getActiveCells(
        self,
    ):
        return self._activeCells.keys()

    def hasChanged(
        self,
    ):
        return True
