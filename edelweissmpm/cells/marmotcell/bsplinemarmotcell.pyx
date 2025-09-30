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
import numpy as np

cimport cython
cimport libcpp.cast
cimport numpy as np
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from edelweissfe.utils.exceptions import CutbackRequest

from libc.stdlib cimport free, malloc
from libcpp.memory cimport allocator, make_unique, unique_ptr

from edelweissmpm.cells.marmotcell.marmotcell cimport (
    MarmotCellFactory,
    MarmotCellWrapper,
    MarmotMaterialPoint,
)
from edelweissmpm.materialpoints.marmotmaterialpoint.mp cimport (
    MarmotMaterialPointWrapper,
)


@cython.final # no subclassing -> cpdef with nogil possible
cdef class BSplineMarmotCellWrapper(MarmotCellWrapper):
    """This class is a wrapper for the MarmotCell class. It is used
    to create a BSpline-MarmotCell object from a set of nodes and knot vectors.
    The MarmotCell object is then used to evaluate the shape functions
    and their derivatives at a given point."""


    def __init__(self, cellType, cellNumber, nodes, knotVectors):
        super().__init__(cellType, cellNumber, nodes)

    def __cinit__(self, str cellType, int cellNumber, list nodes, np.ndarray knotVectors__):
        """This C-level method is responsible for actually creating the MarmotCell.

        Parameters
        ----------
        cellType
            The Marmot cell type which should be represented.
        cellNumber
            The (unique) number of this cell.
        nodes
            The nodes (i.e., control points for this cell.
        knotVectors__
            The set of knotVectors for this cell."""


        self._nodes = nodes
        cdef np.ndarray nodeCoordinates = np.array([ node.coordinates for node in nodes])
        self._nodeCoordinates = nodeCoordinates

        cdef np.ndarray knotVectors_ = knotVectors__
        cdef double[:, ::1] knotVectorsView
        knotVectorsView = knotVectors_

        try:
            self._marmotCell = MarmotCellFactory.createBSplineCell( cellType.encode('utf-8'),
                                                                    self._cellNumber,
                                                                    &self._nodeCoordinates[0,0],
                                                                    self._nodeCoordinates.size,
                                                                    &knotVectorsView[0,0],
                                                                    knotVectorsView.size)

        except ValueError:
            raise NotImplementedError("Marmot cell {:} not found in library.".format(cellType))

    def __dealloc__(self):
        if isinstance(self, BSplineMarmotCellWrapper):
            del self._marmotCell
