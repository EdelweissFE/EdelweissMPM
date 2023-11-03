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
"""
Created on Thu Apr 27 08:35:06 2017

@author: matthias
"""
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
cimport numpy as np
import numpy as np
        
cdef extern from "Marmot/MarmotMPMLibrary.h" namespace "MarmotLibrary" nogil:
    
    cdef cppclass MarmotCellFactory:
        @staticmethod
        MarmotCell* createCell(const string& cellName, 
                               int cellNumber,
                               const double* nodeCoordinates,
                               int sizeNodeCoordinates) except +ValueError

cdef extern from "Marmot/MarmotMaterialPoint.h" nogil:

    cdef cppclass MarmotMaterialPoint:
        pass

cdef extern from "Marmot/MarmotCell.h":
    cdef cppclass MarmotCell nogil:
        
        const vector[vector[string]]& getNodeFields()

        const vector[int]& getDofIndicesPermutationPattern()

        string getCellShape()
        
        int getNNodes()
    
        int getNDofPerCell()

        int isCoordinateInCell(const double *coordinate)

        void assignMaterialPoints(const vector[MarmotMaterialPoint*] materialPoints) 

        void computeMaterialPointKernels(   double* Pe,
                                            double* Ke,
                                            double timeNewTotal,
                                            double dT) except +

        void computeBodyLoad            (   int type, 
                                            const double* load, 
                                            double* Pe,
                                            double* Ke,
                                            double timeNewTotal,
                                            double dT) except +

        const unordered_map[string, int]& getSupportedBodyLoadTypes()

        void interpolateFieldsToMaterialPoints( const double* Q)
        
cdef class MarmotCellWrapper:
    
    cdef MarmotCell* _marmotCell
    cdef list _nodes, 
    cdef int _cellNumber, 
    cdef str _cellType, 
    cdef int _nNodes, _nDof
    cdef list _fields
    cdef str _ensightType
    cdef np.ndarray _dofIndicesPermutation

    cdef dict _supportedBodyLoads
    
    cdef double[:, ::1] _nodeCoordinates

    cpdef void computeMaterialPointKernels(self, 
                     double[::1] residual, 
                     double[::1] dResidual_dQ, 
                     double timeNew, 
                     double dTime, ) nogil

    cpdef void interpolateFieldsToMaterialPoints(self, 
                     double[::1] dQ) nogil 


