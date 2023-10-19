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
"""
Implementing your own cells can be done easily by subclassing from 
the abstract base class :class:`~BaseCell`.
"""

from abc import ABC, abstractmethod
import numpy as np
from fe.points.node import Node
from mpm.cells.base.cell import BaseCell
from mpm.materialpoints.base.mp import BaseMaterialPoint


class Cell(BaseCell):
    def __init__(self, cellType: str, cellNumber: int):
        self._cellType = cellType
        self._cellNumber = cellNumber

    @property
    def cellNumber(self) -> int:
        return self._cellNumber

    @property
    def nNodes(self) -> int:
        return 4

    @property
    def nDof(self) -> int:
        return 4 * 2

    @property
    def fields(self) -> list[list[str]]:
        return [
            [
                "displacement",
            ]
        ] * 4

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        return np.arange(0, 8, dtype=int)

    @property
    def ensightType(self) -> str:
        return "quad4"

    def setNodes(self, nodes: list[Node]):
        self.nodes = nodes
        self.coordinates = np.array([n.coordinates for n in nodes])

        self.x_min = np.min(self.coordinates[:, 0])
        self.x_max = np.max(self.coordinates[:, 0])
        self.y_min = np.min(self.coordinates[:, 1])
        self.y_max = np.max(self.coordinates[:, 1])

    def interpolateSolutionContributionToMaterialPoints(
        self,
        materialPoints: list[BaseMaterialPoint],
        dU: np.ndarray,
    ):
        pass

    def computeMaterialPointKernels(
        self,
        materialPoints: list[BaseMaterialPoint],
        P: np.ndarray,
        K: np.ndarray,
        timeStep: float,
        timeTotal: float,
        dTime: float,
    ):
        pass

    def getCoordinatesAtCenter(self) -> np.ndarray:
        pass

    def isCoordinateInCell(self, coordinate: np.ndarray) -> bool:
        x, y = coordinate

        if x >= self.x_min and x <= self.x_max:
            if y >= self.y_min and y <= self.y_max:
                return True

        return False
