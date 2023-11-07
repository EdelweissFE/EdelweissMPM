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

from mpm.cells.base.cell import BaseCell
from mpm.materialpoints.base.mp import BaseMaterialPoint
from mpm.mpmmanagers.utils import KDTree, Domain, buildEnclosingDomain

import numpy as np


class SmartMaterialPointManager:
    """A smart manager for material points and cells making use of a KDTree for loacation points in cells.

    Parameters
    ----------
    materialPointCells
        The list of BaseCells
    materialPoints
        The list of Materialpoints
    options
        A dictionary containing options
    """

    def __init__(
        self,
        materialPointCells: list[BaseCell],
        materialPoints: list[BaseMaterialPoint],
        options: dict = {"KDTreeLevels": 1},
    ):
        self._cells = materialPointCells
        self._mps = materialPoints
        self._options = options

        self._KDTree = KDTree(
            buildEnclosingDomain(materialPointCells),
            self._options.get("KDTreeLevels"),
            materialPointCells,
        )

    def _checkIfMPPartiallyInCell(self, mp: BaseMaterialPoint, cell: BaseCell):
        """Check if at least one vertex of a MaterialPoint is within a given cell.

        Returns
        -------
        bool
            The truth value if this MaterialPoint is located in the cell.
        """

        for vCoord in mp.getVerticesCoordinates():
            if cell.isCoordinateInCell(vCoord):
                return True

        return False

    def updateConnectivity(
        self,
    ):
        self._activeCells = dict()
        self._attachedMaterialPoints = dict()

        for mp in self._mps:
            currCells = self._KDTree.getCellsForMaterialPoint(mp)
            for cell in currCells:
                if not cell in self._activeCells.keys():
                    self._activeCells[cell] = []
                self._activeCells[cell].append(mp)
        return self._activeCells

    def getActiveCells(
        self,
    ):
        return self._activeCells.keys()

    def getMaterialPointsInCell(self, cell: BaseCell):
        return self._activeCells[cell]

    def hasLostMaterialPoints(
        self,
    ):
        attachedMPs = set([mp for mps in self._activeCells.values() for mp in mps])

        return len(attachedMPs) != len(self._mps)

    def hasChanged(
        self,
    ):
        return True
