#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____ 
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|  
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___ 
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#                                                       
# 
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
# 
#  Matthias Neuner matthias.neuner@uibk.ac.at
# 
#  This file is part of EdelweissFE.
# 
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
# 
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
# Created on Thu Apr 27 08:35:06 2017

# @author: matthias

import numpy as np
cimport numpy as np
cimport libcpp.cast
cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

from fe.utils.exceptions import CutbackRequest
from libcpp.memory cimport unique_ptr, allocator, make_unique
from libc.stdlib cimport malloc, free

from mpm.materialpoints.marmotmaterialpoint.mp cimport MarmotMaterialPointWrapper
from mpm.cells.marmotcell.marmotcell cimport MarmotCellWrapper, MarmotCellFactory, MarmotMaterialPoint
    
@cython.final # no subclassing -> cpdef with nogil possible
cdef class BSplineMarmotCellWrapper(MarmotCellWrapper):

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

        # cdef np.ndarray knotVectors_ = np.array([ knotVector for knotVector in knotVectors])
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

        except IndexError:
            raise NotImplementedError("Marmot cell {:} not found in library.".format(cellType))
