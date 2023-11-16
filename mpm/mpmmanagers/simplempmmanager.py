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

from mpm.cells.base.cell import BaseCell
from mpm.materialpoints.base.mp import BaseMaterialPoint


class SimpleMaterialPointManager:
    """A very simplistic manager for material points and cells.

    Parameters
    ----------
    materialPointCells
        The list of BaseCells
    materialPoints
        The list of Materialpoints
    """

    def __init__(self, materialPointCells: list[BaseCell], materialPoints: list[BaseMaterialPoint]):
        self._cells = materialPointCells
        self._mps = materialPoints

    def _checkIfMPPartiallyInCell(self, mp: BaseMaterialPoint, cell: BaseCell):
        """Check if at least one vertex of a MaterialPoint is within a given cell.

        Returns
        -------
        bool
            The truth value if this MaterialPoint is located in the cell.
        """

        for vCoord in mp.getVertexCoordinates():
            if cell.isCoordinateInCell(vCoord):
                return True

        return False

    def updateConnectivity(
        self,
    ):
        self._activeCells = dict()

        for cell in self._cells:
            mpsInCell = [mp for mp in self._mps if self._checkIfMPPartiallyInCell(mp, cell)]

            if mpsInCell:
                self._activeCells[cell] = mpsInCell

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

        lost = len(attachedMPs) != len(self._mps)

        if lost:
            lost_mps = attachedMPs.symmetric_difference(self._mps)
            for mp in lost_mps:
                print(mp.getVertexCoordinates())

        return lost

    def hasChanged(
        self,
    ):
        return True
