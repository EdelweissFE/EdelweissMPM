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

import numpy as np

from edelweissmpm.cells.base.cell import CellBase
from edelweissmpm.materialpoints.base.mp import MaterialPointBase

cimport numpy as np


cdef int _2pow (int exp) nogil:
    cdef int res = 1 << exp
    return res

cdef class BoundingBox:
    """This class represents a bounding box.

    It is constructed from two points A and B, representing 2 opposing
    vertices of the box.

    Parameters
    ----------
    A
        The first point.
    B
        The opposing point.
    """

    cdef public int dim
    cdef public double[::1] minCoords
    cdef public double[::1] maxCoords

    def __init__(self, double[::1] A, double[::1]B):
        self.dim = len(A)
        self.minCoords = np.minimum(A, B)
        self.maxCoords = np.maximum(A, B)

    cdef int containsCoordinates(self, double[::1] coordinates) nogil:
        cdef int i
        for i in range(self.dim):
            if coordinates[i] <  self.minCoords[i] or coordinates[i] >= self.maxCoords[i]:
                return False

        return True

    cdef double[::1] getCenterCoordinates(self):
        return (np.asarray(self.maxCoords) + np.asarray(self.minCoords)) / 2.0

    cdef double[:,::1] getVertices(self):
        cdef double[:,::1] vertices = np.empty( ( _2pow(  self.dim ), self.dim) )

        cdef int i, j
        for i in range( _2pow(  self.dim )):
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
        cellBoundingBoxMin, cellBoundingBoxMax = cell.getBoundingBox()
        boundingBoxMin

        # for node in cell.nodes:
        for j in range(dimension):
            boundingBoxMin[j] = min ( boundingBoxMin[j], cellBoundingBoxMin[j] )
            boundingBoxMax[j] = max ( boundingBoxMax[j], cellBoundingBoxMax[j] )

    return BoundingBox(boundingBoxMin, boundingBoxMax)

cdef class BoundedCell:
    """This classs represents a tuple of a
    cell and its bounding box.


    Parameters
    ----------
    cell
        The cell.
    boundingBox
        Its BoundingBox instance.
    """

    cdef public BoundingBox boundingBox
    cdef public cell

    def __init__(self, cell, boundingBox):
        self.cell = cell
        self.boundingBox = boundingBox

cdef list filterCellsInBoundingBox(boundedCells, BoundingBox boundingBox ):
    """ Get a list of cells which are overlapping with a given BoundingBox instance.

    Parameters
    ----------
    boundedCells
        The list of BoundedCell instances.
    boundingBox
        The BoundingBox instance for the overlap check.

    Returns
    -------
    list
       The list of BoundedCell instances passing the overlap check.
    """

    cellsInDomain = []

    for cell in boundedCells:
        if (np.asarray(cell.boundingBox.maxCoords) >= np.asarray(boundingBox.minCoords)).all() and (
            (np.asarray(cell.boundingBox.minCoords) < np.asarray(boundingBox.maxCoords)).all()
        ):
            cellsInDomain.append(cell)

    return cellsInDomain

cdef class _KDTreeImpl:
    """This is the background recursive implementation
    of the KDTree branches.

    Parameters
    ----------
    domain
        The domain for this KDTRee level.
    level
        The current level.
    boundedCellsInDomain
        The list of BoundedCell instances which are living in this domain.
    parent
        The parent KDTRee branch. If None, this instance serves as the top level.
    """

    cdef BoundingBox _domain
    cdef double[::1] _centerCoordinates
    cdef int _level
    cdef int _nBoundedCellsInDomain
    cdef dict cellsToChild
    cdef list _children
    cdef _KDTreeImpl _parent
    cdef _boundedCellsInDomain

    def __init__(self, BoundingBox domain , int level, list boundedCellsInDomain, _KDTreeImpl parent = None):
        self._domain = domain
        self._centerCoordinates = self._domain.getCenterCoordinates()
        self._level = level

        self._boundedCellsInDomain = boundedCellsInDomain
        self._nBoundedCellsInDomain = len(self._boundedCellsInDomain)
        self._parent = parent

        if not parent:
            self.cellsToChild = dict()

        if self._level > 0 and self._nBoundedCellsInDomain > _2pow (self._domain.dim):
            self._children = [None] * _2pow(  self._domain.dim )
            for vertice in self._domain.getVertices():
                childDomain = BoundingBox(vertice, self._centerCoordinates)

                self._children[self.getChildIDForCoordinates(vertice)] = _KDTreeImpl(
                    childDomain, self._level - 1,
                    filterCellsInBoundingBox( self._boundedCellsInDomain, childDomain ),
                    self
                )
        else:
            self.linkBoundedCellsToChild(self._boundedCellsInDomain, self)

    cdef getCellForCoordinatesUpwards(self, double[::1] coordinates):
        """Search for a recursively cell by first going levels upwards until
        the domain covers the coordinate, followed by a downward search.

        Parameters
        ----------
        coordinates
            The coordinates in question.

        Returns
        -------
        CellBase
            The found cell.
        """

        if self._domain.containsCoordinates(coordinates):
            return self.getCellForCoordinatesDownwards(coordinates)

        return self._parent.getCellForCoordinatesUpwards(coordinates)

    cdef getCellForCoordinatesDownwards(self, double[::1] coordinates):
        """Search for a recursively cell by first going levels down until
        until the lowest level is reached. Then, all covered cells are asked for
        the requested coordinates.

        Parameters
        ----------
        coordinates
            The coordinates in question.

        Returns
        -------
        CellBase
            The found cell.
        """

        if self._children:
            return (<_KDTreeImpl>self._children[self.getChildIDForCoordinates(coordinates)]).getCellForCoordinatesDownwards(coordinates)

        else:
            for cell in self._boundedCellsInDomain:
                if cell.cell.isCoordinateInCell(np.asarray(coordinates)):
                    return cell.cell

        raise Exception("Failed to determine cell for coordinate {:}".format(coordinates))

    cdef int getChildIDForCoordinates(self, double[::1] coordinates ) nogil:
        """Compute the sub-divison ID for the K childs containing the given coordinates.

        Parameters
        ----------
        coordinates
            The coordinates in question.

        Returns
        -------
        int
            The ID of the child containing the requested coordinates.
        """

        cdef int num = 0
        cdef int i
        for i in range(self._domain.dim):
            if coordinates[i] < self._centerCoordinates[i]:
                num += _2pow(i)
        return num

    cdef linkBoundedCellsToChild(self, cells, child):
        """For fast access of lowest level KDTree childs for a given cell,
        The lowest level instances communicate their covered cells up to
        the top level parent instance.

        Parameters
        ----------
        cells
            The covered cells.
        child
            The lowest level KDTree instance covering those cells.
        """

        if self._parent:
            return self._parent.linkBoundedCellsToChild(cells, child)

        for cell in cells:
            self.cellsToChild[cell.cell] = child


cdef class KDTree:
    """This is an efficient Kd tree implementation for searching
    the correct cell for a given (material point) coordinate.

    Parameters
    ----------
    domain
        The domain which should be covered.
    level
        The maximum number of kd tree levels.
    cellsInDomain
        The list of cells in this domain.
    """

    cdef BoundingBox _domain
    cdef _KDTreeImpl _tree
    cdef _cellsInDomain

    def __init__(self, domain , int level, cellsInDomain):
        self._domain = domain
        self._cellsInDomain = cellsInDomain
        boundedCells = [ BoundedCell(cell, buildBoundingBoxFromCells( [ cell, ], self._domain.dim )) for cell in cellsInDomain ]

        self._tree = _KDTreeImpl( self._domain, level, boundedCells, None)

    def getCellForCoordinates(self, double[::1] coordinates, list initialGuess=None) -> CellBase:
        """Efficient lookup for the cell which contains the given coordinate.
        If a list of initial guess cells is provided, the lookup speed can be increased
        drastically.

        Parameters
        ----------
        coordinates
            The coordinates to be found.
        initialGuess
            The optional list of cells for improved lookup speed.

        Returns
        -------
        CellBase
            The cell containing the coordinate.
        """

        if not self._domain.containsCoordinates(coordinates):
            raise Exception("requested coordinate {:} outside domain".format(np.asarray(coordinates)))

        if initialGuess:
            for cell in initialGuess:
                if cell.isCoordinateInCell(np.asarray(coordinates)):
                    return cell

            # that failed, but let's check the parent KDTree of the first initial guess cell.
            return (<_KDTreeImpl>self._tree.cellsToChild[initialGuess[0]]).getCellForCoordinatesUpwards(coordinates)

        return self._tree.getCellForCoordinatesDownwards(coordinates)
