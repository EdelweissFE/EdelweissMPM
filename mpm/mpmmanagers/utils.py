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

import numpy as np


class BoundingBox:
    def __init__(self, A: np.ndarray, B: np.ndarray):
        self.dim = len(A)
        self._minCoords = np.minimum(A, B)
        self._maxCoords = np.maximum(A, B)

    @property
    def minCoords(self):
        return self._minCoords

    @property
    def maxCoords(self):
        return self._maxCoords

    def getCenterCoordinates(self):
        return (self._maxCoords + self._minCoords) / 2.0

    def getVertices(self):
        vertices = []

        vertices.append([self._minCoords[0]])
        vertices.append([self._maxCoords[0]])

        if self.dim > 1:
            vertices[0].append(self._minCoords[1])
            vertices[1].append(self._minCoords[1])
            vertices.append([self._minCoords[0], self._maxCoords[1]])
            vertices.append([self._maxCoords[0], self._maxCoords[1]])

            if self.dim > 2:
                vertices[0].append(self._minCoords[2])
                vertices[1].append(self._minCoords[2])
                vertices[2].append(self._minCoords[2])
                vertices[3].append(self._minCoords[2])
                vertices.append([self._minCoords[0], self._minCoords[1], self._maxCoords[2]])
                vertices.append([self._maxCoords[0], self._minCoords[1], self._maxCoords[2]])
                vertices.append([self._minCoords[0], self._maxCoords[1], self._maxCoords[2]])
                vertices.append([self._maxCoords[0], self._maxCoords[1], self._maxCoords[2]])

        return vertices

    def __str__(self):
        string = "BoundingBox\n"
        string += " min coords = " + str(self._minCoords) + "\n"
        string += " max coords = " + str(self._maxCoords) + "\n"
        return string


def buildBoundingBoxFromCells(materialPointCells: list[CellBase], dimension: int = 2):
    cellsVertices = np.array([np.array([n.coordinates for n in cell.nodes]) for cell in materialPointCells])

    # min/max over all cells and nodes
    boundingBoxMin = np.min(cellsVertices, axis=(0, 1))
    boundingBoxMax = np.max(cellsVertices, axis=(0, 1))

    return BoundingBox(boundingBoxMin, boundingBoxMax)


class KDTree:
    def __init__(self, domain: BoundingBox, level: int, potentialCells: set[CellBase]):
        self._domain = domain
        self._dimension = domain.dim
        self._centerCoordinates = domain.getCenterCoordinates()
        self._level = level

        self._children = dict()

        self._cellsInDomain = self.filterCellsInDomain(potentialCells)

        self.buildTree()

    def buildTree(self):
        if self._level > 0:
            for vertice in self._domain.getVertices():
                childDomain = BoundingBox(vertice, self._centerCoordinates)

                self._children[self.getChildIDForCoordinates(vertice)] = KDTree(
                    childDomain, self._level - 1, self._cellsInDomain
                )

    def filterCellsInDomain(self, cells):
        cellsInDomain = []

        for cell in cells:
            cellBoundingBox = buildBoundingBoxFromCells(
                [
                    cell,
                ]
            )

            if ((cellBoundingBox.maxCoords >= self._domain.minCoords).all()).all() and (
                (cellBoundingBox.minCoords < self._domain.maxCoords).all()
            ):
                cellsInDomain.append(cell)

        return cellsInDomain

    def getCellForCoordinates(self, coordinates: np.ndarray):
        if self._children:
            return self._children[self.getChildIDForCoordinates(coordinates)].getCellForCoordinates(coordinates)

        else:
            for cell in self._cellsInDomain:
                if cell.isCoordinateInCell(coordinates):
                    return cell

        raise Exception("Failed to determine cell for coordinate {:}".format(coordinates))

    def getChildIDForCoordinates(self, coordinates: np.ndarray):
        num = 0
        for i in range(len(coordinates)):
            if coordinates[i] < self._centerCoordinates[i]:
                num += 2**i
        return num
