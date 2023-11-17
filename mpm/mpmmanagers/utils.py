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

from mpm.cells.base.cell import BaseCell
from mpm.materialpoints.base.mp import BaseMaterialPoint

import numpy as np


class BoundingBox:
    def __init__(self, A: list, B: list):
        self.dim = len(A)
        self.initializeFromTwoPoints(A, B)

    @property
    def minCoords(self):
        return self._minCoords

    @property
    def maxCoords(self):
        return self._maxCoords

    def getCenterCoordinates(self):
        centercoords = []
        for i in range(self.dim):
            centercoords.append((self._minCoords[i] + self._maxCoords[i]) / 2.0)

        return centercoords

    def getMinCoordinates(self):
        return self._minCoords

    def getMaxCoordinates(self):
        return self._maxCoords

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

    def initializeFromTwoPoints(self, A: list, B: list):
        self._minCoords = []
        self._maxCoords = []

        for i in range(self.dim):
            self._minCoords.append(min(A[i], B[i]))
            self._maxCoords.append(max(A[i], B[i]))

    def isInside(self, point: list):
        for i in range(self.dim):
            if point[i] > self._maxCoords[i] or point[i] < self._minCoords[i]:
                return False

        return True

    def __str__(self):
        string = "BoundingBox\n"
        string = " min coords = " + str(self._minCoords) + "\n"
        string = " max coords = " + str(self._maxCoords) + "\n"
        return string


def getBoundingBoxForCell(cell):
    coordinates = np.array([n.coordinates for n in cell.nodes])

    boundingBoxMin = []
    boundingBoxMax = []

    for i in range(len(coordinates[0, :])):
        boundingBoxMin.append(min(coordinates[:, i]))
        boundingBoxMax.append(max(coordinates[:, i]))

    return boundingBoxMin, boundingBoxMax


def buildModelBoundingBox(materialPointCells: list[BaseCell], dimension: int = 2):
    # initialize with first cell
    A, B = getBoundingBoxForCell(materialPointCells[0])

    d = BoundingBox(A, B)

    for cell in materialPointCells[1:]:
        boundingBoxCurrCellMin, boundingBoxCurrCellMax = getBoundingBoxForCell(cell)

        for i in range(d.dim):
            d.minCoords[i] = min(d.minCoords[i], boundingBoxCurrCellMin[i])
            d.maxCoords[i] = max(d.minCoords[i], boundingBoxCurrCellMax[i])

    return d


class KDTree:
    def __init__(self, domain: BoundingBox, level: int, materialPointCells: list[BaseCell]):
        self._domain = domain
        self._dimension = domain.dim
        self._cells = materialPointCells
        self._level = level

        self._hasChildren = False
        self._children = []

        self.buildTree()

        self._cellsInDomain = self.assignCellsToDomain()

    def buildTree(self):
        if self._level > 0:
            self._hasChildren = True
            centerCoordinates = self._domain.getCenterCoordinates()

            for vertice in self._domain.getVertices():
                newDomain = BoundingBox(vertice, centerCoordinates)

                self._children.append(KDTree(newDomain, self._level - 1, self._cells))

    def assignCellsToDomain(self):
        cellsInDomain = []

        for cell in self._cells:
            boundingBoxCurrCellMin, boundingBoxCurrCellMax = getBoundingBoxForCell(cell)

            appendCell = True
            for i in range(self._domain.dim):
                if (
                    boundingBoxCurrCellMax[i] < self._domain.minCoords[i]
                    or boundingBoxCurrCellMin[i] > self._domain.maxCoords[i]
                ):
                    appendCell = False
                    break

            if appendCell:
                cellsInDomain.append(cell)

        return cellsInDomain

    def getCellForCoordinates(self, coordinates):
        if self._hasChildren:
            for child in self._children:
                if child._domain.isInside(coordinates):
                    return child.getCellForCoordinates(coordinates)
        else:
            for cell in self._cellsInDomain:
                if cell.isCoordinateInCell(coordinates):
                    return cell

    def getCellsForMaterialPoint(self, mp: BaseMaterialPoint):
        activeCells = []

        if self._hasChildren:
            for vCoord in mp.getVertexCoordinates():
                for child in self._children:
                    if child._domain.isInside(vCoord):
                        activeCells.append(child.getCellForCoordinates(vCoord))

        else:
            for vCoord in mp.getVerticesCoordinates():
                for cell in self._cellsInDomain:
                    if cell.isCoordinateInCell(vCoord):
                        activeCells.append(cell)

        return activeCells
