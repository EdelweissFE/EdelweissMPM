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

cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np


cdef extern from "Marmot/Marmot.h" namespace "MarmotLibrary" nogil:
    cdef cppclass MarmotMaterialFactory:
        @staticmethod
        int getMaterialCodeFromName(const string& materialName) except +ValueError

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

        void assignMaterial( const MarmotMaterialSection& property ) except +

        void initializeYourself()

        void prepareYourself(double timeNew, double dT)

        void computeYourself(double timeNew, double dT) except +

        void acceptStateAndPosition()

        StateView getStateView( const string& stateName )

        int getDimension()

        int getNumberOfVertices()

        void getVertexCoordinates(double* )

        void getCenterCoordinates(double* )

        double getVolume()

cdef class MarmotMaterialPointWrapper:

    cdef MarmotMaterialPoint* _marmotMaterialPoint
    cdef int _number,
    cdef str _materialPointType,
    cdef str _ensightType
    cdef int _nVertices

    cdef public double[::1] _stateVars
    cdef public double[::1] _stateVarsTemp
    cdef int _nStateVars
    cdef int _nDim

    cdef public list _assignedCells

    cdef public double[::1]  _materialProperties

    cdef public np.ndarray _centerCoordinates
    cdef public np.ndarray _vertexCoordinates

    cdef double[::1] _centerCoordinatesView
    cdef double[:,::1] _vertexCoordinatesView

    # nogil methods are already declared here

    cpdef void _initializeStateVarsTemp(self, ) nogil

    cpdef void computeYourself(self,
                     double timeNew,
                     double dTime, ) except * nogil

    cdef double[::1] getStateView(self, string stateName)
