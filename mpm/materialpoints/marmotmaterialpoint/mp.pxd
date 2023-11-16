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
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np

cdef extern from "Marmot/Marmot.h" namespace "MarmotLibrary" nogil:
    cdef cppclass MarmotMaterialFactory:
        @staticmethod
        int getMaterialCodeFromName(const string& materialName) except +IndexError

cdef extern from "Marmot/MarmotUtils.h":
    cdef struct StateView:
        double *stateLocation
        int stateSize

cdef extern from "Marmot/MarmotElementProperty.h":
    cdef cppclass MarmotElementProperty nogil:
        pass
    
    cdef cppclass MarmotMaterialSection(MarmotElementProperty) nogil:
        MarmotMaterialSection(int materialCode, const double* materialProperties, int nMaterialProperties)
        
cdef extern from "Marmot/MarmotMPMLibrary.h" namespace "MarmotLibrary" nogil:
    
    cdef cppclass MarmotMaterialPointFactory:
        @staticmethod
        MarmotMaterialPoint* createMaterialPoint(const string& materialPointName, 
                                                 int materialPointNumber,
                                                 const double* vertexCoordinates,
                                                 int sizeVertexCoordinates,
                                                 double volume
                                                 ) except +ValueError

        
cdef extern from "Marmot/MarmotMaterialPoint.h":
    cdef cppclass MarmotMaterialPoint nogil:

        int getNumberOfRequiredStateVars() 

        string getMaterialPointShape() 

        void assignStateVars( double* stateVars, int nStateVars ) 

        void assignMaterial( const MarmotMaterialSection& property ) except +ValueError

        void initializeYourself() 

        void prepareYourself(double timeNew, double dT) 

        void computeYourself(double timeNew, double dT) except +RuntimeError

        StateView getStateView( const string& stateName ) 

        # void getCoordinatesAtCenter(double* ) 

        int getDimension()

        int getNumberOfVertices() 

        # void assignVertexCoordinates(const double* ) 

        void getVertexCoordinates(double* ) 

        double getVolume() 
        
cdef class MarmotMaterialPointWrapper:
    
    cdef MarmotMaterialPoint* _marmotMaterialPoint
    cdef int _label, 
    cdef str _materialPointType, 
    cdef str _ensightType
    cdef int _nVertices
    
    cdef int _hasMaterial
    cdef public double[::1] _stateVars
    cdef public double[::1] _stateVarsTemp  
    cdef int _nStateVars
    cdef int _nDim 

    cdef public double[::1]  _materialProperties

    cdef public np.ndarray _centerCoordinates
    cdef public np.ndarray _vertexCoordinates
    
    cdef double[::1] _centerCoordinatesView
    cdef double[:,::1] _vertexCoordinatesView

    # nogil methods are already declared here

    cpdef void _initializeStateVarsTemp(self, ) nogil

    cpdef void computeYourself(self, 
                     double timeNew, 
                     double dTime, ) nogil except *

    cdef double[::1] getStateView(self, string stateName)