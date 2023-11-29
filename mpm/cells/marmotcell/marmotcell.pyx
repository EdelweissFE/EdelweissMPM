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

import numpy as np
cimport numpy as np
cimport mpm.cells.marmotcell.marmotcell
cimport libcpp.cast
cimport cython

from fe.utils.exceptions import CutbackRequest
from libcpp.memory cimport unique_ptr, allocator, make_unique
from libc.stdlib cimport malloc, free

from mpm.materialpoints.marmotmaterialpoint.mp cimport MarmotMaterialPointWrapper
    
@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotCellWrapper:
    """This cell as a wrapper for MarmotCells.

    For the documentation of MarmotCells, please refer to `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.

    Parameters
    ----------
    cellType
        The Marmot element which should be represented, e.g., CPE4.
    cellNumber
        The (unique) label of this cell.
    nodes
        The list of nodes for this Cell.
        """


    def __init__(self, cellType, cellNumber, nodes):
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

    @property
    def cellNumber(self):
        return self._cellNumber
    
    @property
    def cellType(self):
        return self._cellType

    @property
    def nodes(self):
        return self._nodes

    @property
    def nNodes(self):
        return self._nNodes
    
    @property
    def nDof(self):
        return self._nDof

    @property
    def fields(self):
        return self._fields

    @property
    def dofIndicesPermutation(self):
        return self._dofIndicesPermutation

    @property
    def ensightType(self):
        return self._ensightType

    @property
    def assignedMaterialPoints(self):
        return self._assignedMaterialPoints

    cpdef void computeMaterialPointKernels(self, 
                         double[::1] dUc, 
                         double[::1] Pc, 
                         double[::1] Kc, 
                         double timeNew, 
                         double dTime, ) nogil:
        """Evaluate residual and stiffness for given time, field, and field increment."""

        self._marmotCell.computeMaterialPointKernels(&dUc[0], &Pc[0], &Kc[0], timeNew, dTime)

    cpdef void interpolateFieldsToMaterialPoints(self, double[::1] dUc) nogil:

        self._marmotCell.interpolateFieldsToMaterialPoints(&dUc[0])

    def computeBodyLoad(self, 
                         str loadType, 
                         double[::1] load, 
                         double[::1] Pe, 
                         double[::1] Ke, 
                         double timeNew, 
                         double dTime, ):
        self._marmotCell.computeBodyLoad( self._supportedBodyLoads[loadType.upper()], &load[0], &Pe[0], &Ke[0], timeNew, dTime)

    def assignMaterialPoints(self, list marmotMaterialPointWrappers):
        cdef vector[MarmotMaterialPoint*] mps
        cdef MarmotMaterialPointWrapper mpWrapper

        for mpWrapper in marmotMaterialPointWrappers:
            mps.push_back(<MarmotMaterialPoint*> mpWrapper._marmotMaterialPoint)

        self._marmotCell.assignMaterialPoints(mps)

        self._assignedMaterialPoints = marmotMaterialPointWrappers
        
    def isCoordinateInCell(self, coordinate: np.ndarray) -> bool:
        cdef double[::1] coords = coordinate
        return self._marmotCell.isCoordinateInCell(&coords[0])

    def getInterpolationVector(self, coordinate: np.ndarray) -> np.ndarray:
        cdef double[::1] coords = coordinate
        cdef np.ndarray N = np.zeros(self._nNodes)
        cdef double[::1] Nview_ = N
        self._marmotCell.getInterpolationVector(&Nview_[0], &coords[0])
        return N
    
    def __dealloc__(self):
        del self._marmotCell