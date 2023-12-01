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
cimport numpy as np


cdef class BoundingBox:

    cdef public int dim
    cdef public double[::1] minCoords
    cdef public double[::1] maxCoords

    def __init__(self, double[::1] A, double[::1]B):
        self.dim = len(A)
        self.minCoords = np.minimum(A, B)
        self.maxCoords = np.maximum(A, B)

    cdef int containsCoordinates(self, double[::1] coordinates):
        cdef int i
        for i in range(self.dim):
            if coordinates[i] <  self.minCoords[i] or coordinates[i] >= self.maxCoords[i]:
                return False

        return True
    
    cdef double[::1] getCenterCoordinates(self):
        return (np.asarray(self.maxCoords) + np.asarray(self.minCoords)) / 2.0

    cdef double[:,::1] getVertices(self):
        cdef double[:,::1] vertices = np.empty( ( 2 ** self.dim, self.dim) )

        cdef int i, j
        for i in range( 2 ** self.dim):
            for j in range(self.dim):

                vertices[i, j] = self.minCoords[j] if (i & (1 << j) ) else self.maxCoords[j] 

        return vertices

    def __str__(self):
        string = "BoundingBox\n"
        string += " min coords = " + str(self._minCoords) + "\n"
        string += " max coords = " + str(self._maxCoords) + "\n"
        return string

def buildBoundingBoxFromCells(materialPointCells, int dimension):

    cdef double[::1] boundingBoxMin = np.full(dimension, 1e16)
    cdef double[::1] boundingBoxMax = np.full(dimension, -1e16)

    cdef int j

    for cell in materialPointCells:
        for node in cell.nodes:
            for j in range(dimension):
                boundingBoxMin[j] = min ( boundingBoxMin[j], node.coordinates[j])
                boundingBoxMax[j] = max ( boundingBoxMax[j], node.coordinates[j])

    return BoundingBox(boundingBoxMin, boundingBoxMax)


cdef list filterCellsInBoundingBox(cells, BoundingBox boundingBox, cellsBoundingBoxes):
    cellsInDomain = []

    for cell in cells:
        cellBoundingBox = cellsBoundingBoxes[cell]

        if (np.asarray(cellBoundingBox.maxCoords) >= np.asarray(boundingBox.minCoords)).all() and (
            (np.asarray(cellBoundingBox.minCoords) < np.asarray(boundingBox.maxCoords)).all()
        ):
            cellsInDomain.append(cell)

    return cellsInDomain

cdef class KDTree:

    cdef BoundingBox _domain
    cdef int _dimension
    cdef double[::1] _centerCoordinates
    cdef int _level
    cdef int _nCellsInDomain
    cdef dict cellsToChild
    cdef list _children
    cdef KDTree _parent
    cdef _cellsInDomain

    def __init__(self, domain , int level, cellsInDomain, KDTree parent = None, cellsBoundingBoxes = None):
        self._domain = domain
        self._dimension = domain.dim
        self._centerCoordinates = self._domain.getCenterCoordinates()
        self._level = level

        self._cellsInDomain = cellsInDomain

        self._nCellsInDomain = len(self._cellsInDomain)

        self._parent = parent

        if not parent:
            self.cellsToChild = dict()
            cellsBoundingBoxes = {cell:buildBoundingBoxFromCells( [ cell, ], self._domain.dim ) for cell in cellsInDomain }

        self.buildTree(cellsBoundingBoxes)

    cdef buildTree(self, cellsBoundingBoxes):
        if self._level > 0 and self._nCellsInDomain > 2 ** self._dimension :
            self._children = [None] * 2**self._dimension
            for vertice in self._domain.getVertices():
                childDomain = BoundingBox(vertice, self._centerCoordinates)

                self._children[self.getChildIDForCoordinates(vertice)] = KDTree(
                    childDomain, self._level - 1, 
                    filterCellsInBoundingBox( self._cellsInDomain, childDomain, cellsBoundingBoxes),
                    self,
                    cellsBoundingBoxes
                )
        else:
            self.setCellsToChild(self._cellsInDomain, self)

    cpdef getCellForCoordinates(self, double[::1] coordinates, list initialGuess=None):
    
        if initialGuess:
            for cell in initialGuess:
                if cell.isCoordinateInCell(np.asarray(coordinates)):
                    return cell
            
            # that failed, but let's check the parent KDTree of the first initial guess cell.
            return (<KDTree>self.cellsToChild[initialGuess[0]]).getCellForCoordinatesUpwards(coordinates)

        return self.getCellForCoordinatesDownwards(coordinates)

    cdef getCellForCoordinatesUpwards(self, double[::1] coordinates):

        if self._domain.containsCoordinates(coordinates):
            return self.getCellForCoordinatesDownwards(coordinates)

        return self._parent.getCellForCoordinatesUpwards(coordinates)

    cdef getCellForCoordinatesDownwards(self, double[::1] coordinates):

        if self._children:
            return (<KDTree>self._children[self.getChildIDForCoordinates(coordinates)]).getCellForCoordinatesDownwards(coordinates)

        else:
            for cell in self._cellsInDomain:
                if cell.isCoordinateInCell(np.asarray(coordinates)):
                    return cell

        raise Exception("Failed to determine cell for coordinate {:}".format(coordinates))

    cdef int getChildIDForCoordinates(self, double[::1] coordinates ):
        cdef int num = 0
        cdef int i 
        for i in range(self._dimension):
            if coordinates[i] < self._centerCoordinates[i]:
                num += 2**i
        return num

    cdef setCellsToChild(self, cells, child):
    
        if self._parent:
            return self._parent.setCellsToChild(cells, child) 

        for cell in cells: 
            self.cellsToChild[cell] = child

