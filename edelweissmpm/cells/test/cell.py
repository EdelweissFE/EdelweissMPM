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
the abstract base class :class:`~CellBase`.
"""


import numpy as np
from edelweissfe.points.node import Node

from edelweissmpm.cells.base.cell import CellBase
from edelweissmpm.materialpoints.base.mp import MaterialPointBase


class Cell(CellBase):
    def __init__(self, cellType: str, cellNumber: int, nodes: list[Node]):
        self._cellType = cellType
        self._cellNumber = cellNumber
        self._assignedMaterialPoints = list()

        self.nodes = nodes
        self.coordinates = np.array([n.coordinates for n in nodes])

        self.x_min = np.min(self.coordinates[:, 0])
        self.x_max = np.max(self.coordinates[:, 0])
        self.y_min = np.min(self.coordinates[:, 1])
        self.y_max = np.max(self.coordinates[:, 1])

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

    @property
    def assignedMaterialPoints(self) -> list:
        return self._assignedMaterialPoints

    def assignMaterialPoints(self, materialPoints):
        self.materialPoints = materialPoints

    def interpolateSolutionContributionToMaterialPoints(
        self,
        materialPoints: list[MaterialPointBase],
        dU: np.ndarray,
    ):
        pass

    def computeMaterialPointKernels(
        self,
        materialPoints: list[MaterialPointBase],
        P: np.ndarray,
        K: np.ndarray,
        timeStep: float,
        timeTotal: float,
        dTime: float,
    ):
        pass

    def computeBodyLoad(
        self,
        loadType: str,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        pass

    def computeDistributedLoad(
        self,
        loadType: str,
        surfaceID: int,
        materialPoint: MaterialPointBase,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        pass

    def getCoordinatesAtCenter(self) -> np.ndarray:
        pass

    def getBoundingBox(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.x_min, self.y_min]), np.array([self.x_max, self.y_max]))

    def isCoordinateInCell(self, coordinate: np.ndarray) -> bool:
        x, y = coordinate

        if x >= self.x_min and x <= self.x_max:
            if y >= self.y_min and y <= self.y_max:
                return True

        return False

    def getInterpolationVector(self, coordinate) -> np.ndarray:
        raise Exception("not implemented!")
