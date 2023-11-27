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
cdef class LagrangianMarmotCellWrapper(MarmotCellWrapper):
    # cdef classes cannot subclass. Hence we do not subclass from the BaseCell,
    # but still we follow the interface for compatiblity.

    def __init__(self, cellType, cellNumber, nodes):
        """This element serves as a wrapper for LagrangianMarmotCells.

        For the documentation of MarmotCells, please refer to `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.

        Parameters
        ----------
        elementType 
            The Marmot element which should be represented, e.g., CPE4.
        elNumber
            The number of the element."""

            
        self._cellNumber = cellNumber
        self._cellType = cellType
        
        self._nNodes                         = self._marmotCell.getNNodes()
        
        self._nDof                           = self._marmotCell.getNDofPerCell()
        
        cdef vector[vector[string]] fields  = self._marmotCell.getNodeFields()
        self._fields                         = [ [ s.decode('utf-8')  for s in n  ] for n in fields ]
        
        cdef vector[int] permutationPattern = self._marmotCell.getDofIndicesPermutationPattern()
        self._dofIndicesPermutation          = np.asarray(permutationPattern)
    
        cdef dict supportedBodyLoads = self._marmotCell.getSupportedBodyLoadTypes()
        self._supportedBodyLoads = {k.decode() :  v for k, v in supportedBodyLoads.items()  }
        
        self._ensightType                    = self._marmotCell.getCellShape().decode('utf-8')

    def __cinit__(self, str cellType, int cellNumber, list nodes):
        """This C-level method is responsible for actually creating the MarmotCell.

        Parameters
        ----------
        elementType 
            The Marmot element which should be represented, e.g., CPE4.
        elNumber
            The number of the element."""


        self._nodes = nodes
        cdef np.ndarray nodeCoordinates = np.array([ node.coordinates for node in nodes])
        self._nodeCoordinates = nodeCoordinates

        try:
            self._marmotCell = MarmotCellFactory.createCell( cellType.encode('utf-8'), self._cellNumber, &self._nodeCoordinates[0,0], self._nodeCoordinates.size)
        except IndexError:
            raise NotImplementedError("Marmot cell {:} not found in library.".format(cellType))
