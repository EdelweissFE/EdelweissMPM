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


class Domain:
    x_max = None
    y_max = None
    z_max = None
    x_min = None
    y_min = None
    z_min = None

    def __init__(self, dimension: int = 2):
        self.dim = dimension

    def getCenterCoordinates(self):
        centercoords = [(self.x_min + self.x_max) / 2.0]

        if self.dim > 1:
            centercoords.append((self.y_min + self.y_max) / 2.0)

            if self.dim > 2:
                centercoords.append((self.z_min + self.z_max) / 2.0)

        return centercoords

    def setCoordinates(self, minCoords: list, maxCoords: list):
        self.x_min = minCoords[0]
        self.x_max = maxCoords[0]

        if self.dim > 1:
            self.y_min = minCoords[1]
            self.y_max = maxCoords[1]

            if self.dim > 2:
                self.z_min = minCoords[2]
                self.z_max = maxCoords[2]

    def getMinCoordinates(self):
        return list((self.x_min, self.y_min, self.z_min))

    def getMaxCoordinates(self):
        return list((self.x_max, self.y_max, self.z_max))

    def getVertices(self):
        vertices = []
        vertices.append(self.getMinCoordinates())
        vertices.append([self.x_max, self.y_min, self.z_min])

        if self.dim > 1:
            vertices.append([self.x_min, self.y_max, self.z_min])
            vertices.append([self.x_max, self.y_max, self.z_min])

            if self.dim > 2:
                vertices.append([self.x_min, self.y_min, self.z_max])
                vertices.append([self.x_max, self.y_min, self.z_max])
                vertices.append([self.x_min, self.y_max, self.z_max])
                vertices.append([self.x_max, self.y_max, self.z_max])
        return vertices

    def createFromTwoPoints(self, A: list, B: list):

        self.x_min = min(A[0], B[0])
        self.x_max = max(A[0], B[0])

        if self.dim > 1:
            self.y_min = min(A[1], B[1])
            self.y_max = max(A[1], B[1])

            if self.dim > 2:
                self.z_min = min(A[2], B[2])
                self.z_max = max(A[2], B[2])

    def isInside(self, point: list):

        x = point[0]

        if x > self.x_max or x < self.x_min:
            return False
        if self.dim > 1:
            y = point[1]
            if y > self.y_max or y < self.y_min:
                return False
            if self.dim > 2:
                z = point[2]
                if z > self.z_max or z < self.z_min:
                    return False

        return True

    def __str__(self):
        string = "Domain\n"
        string += " ( (x_min, x_max) = ({:},{:})\n".format(self.x_min, self.x_max)
        string += "   (y_min, y_max) = ({:},{:})\n".format(self.y_min, self.y_max)
        string += "   (z_min, z_max) = ({:},{:}))".format(self.z_min, self.z_max)
        return string


def buildEnclosingDomain(materialPointCells: list[BaseCell], dimension: int = 2):

    d = Domain(dimension)
    firstCell = materialPointCells[1]

    # initialize with first cell
    d.x_min = firstCell.x_min
    d.x_max = firstCell.x_max

    if dimension > 1:
        d.y_min = firstCell.y_min
        d.y_max = firstCell.y_max

        if dimension > 2:
            d.z_min = firstCell.z_min
            d.z_max = firstCell.z_max

    for cell in materialPointCells[1:]:

        d.x_min = cell.x_min if cell.x_min < d.x_min else d.x_min
        d.x_max = cell.x_max if cell.x_max > d.x_max else d.x_max
        if dimension > 1:
            d.y_min = cell.y_min if cell.y_min < d.y_min else d.y_min
            d.y_max = cell.y_max if cell.y_max > d.y_max else d.y_max
            if dimension > 2:
                d.z_min = cell.z_min if cell.z_min < d.z_min else d.z_min
                d.z_max = cell.z_max if cell.z_max > d.z_max else d.z_max

    return d


class KDTree:
    def __init__(self, domain: Domain, level: int, materialPointCells: list[BaseCell]):
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

                newDomain = Domain(self._dimension)
                newDomain.createFromTwoPoints(vertice, centerCoordinates)

                self._children.append(KDTree(newDomain, self._level - 1, self._cells))

    def assignCellsToDomain(self):

        cellsInDomain = []

        for cell in self._cells:

            if cell.x_max < self._domain.x_min or cell.x_min > self._domain.x_max:
                continue

            if self._dimension > 1:
                if cell.y_max < self._domain.y_min or cell.y_min > self._domain.y_max:
                    continue

                    if self._dimension > 2:
                        if (
                            cell.z_max < self._domain.z_min
                            or cell.z_min > self._domain.z_max
                        ):
                            continue

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
