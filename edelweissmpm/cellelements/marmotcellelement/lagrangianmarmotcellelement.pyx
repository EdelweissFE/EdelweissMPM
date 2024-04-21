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

from edelweissfe.utils.exceptions import CutbackRequest
from libcpp.memory cimport unique_ptr, allocator, make_unique
from libc.stdlib cimport malloc, free

from edelweissmpm.materialpoints.marmotmaterialpoint.mp cimport MarmotMaterialPointWrapper
from edelweissmpm.cellelements.marmotcellelement.marmotcellelement cimport MarmotCellElementWrapper, MarmotCellElementFactory, MarmotMaterialPoint, MarmotCell, MarmotCellElement
    
@cython.final # no subclassing -> cpdef with nogil possible
cdef class LagrangianMarmotCellElementWrapper(MarmotCellElementWrapper):

    def __init__(self, cellType, cellNumber, gridnodes):
        super().__init__(cellType, cellNumber, gridnodes)

    def __cinit__(self, str cellType, int cellNumber, list gridnodes):
        """This C-level method is responsible for actually creating the MarmotCellElement.

        Parameters
        ----------
        elementType 
            The Marmot element which should be represented, e.g., CPE4.
        elNumber
            The number of the element."""


        self._gridnodes = gridnodes
        cdef np.ndarray gridnodeCoordinates = np.array([ gridnode.coordinates for gridnode in gridnodes])
        self._gridnodeCoordinates = gridnodeCoordinates

        try:
            self._marmotCell = self._marmotCellElement = MarmotCellElementFactory.createCellElement( cellType.encode('utf-8'), self._cellNumber, &self._gridnodeCoordinates[0,0], self._gridnodeCoordinates.size)
        except IndexError:
            raise NotImplementedError("Marmot cellelement {:} not found in library.".format(cellType))
    
    def __dealloc__(self):
        if type(self) is LagrangianMarmotCellElementWrapper:
            del self._marmotCellElement
