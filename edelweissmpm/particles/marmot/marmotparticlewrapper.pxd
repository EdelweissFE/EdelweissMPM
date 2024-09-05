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
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np

from edelweissmpm.materialpoints.marmotmaterialpoint.mp cimport MarmotMaterialPoint
from edelweissmpm.meshfree.approximations.marmot.marmotmeshfreeapproximation cimport (
    MarmotMeshfreeApproximation,
)
from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction cimport (
    MarmotMeshfreeKernelFunction,
)


cdef extern from "Marmot/Marmot.h" namespace "MarmotLibrary" nogil:
    cdef cppclass MarmotMaterialFactory:
        @staticmethod
        int getMaterialCodeFromName(const string& materialName) except +ValueError

cdef extern from "Marmot/MarmotUtils.h":
    cdef struct StateView:
        double *stateLocation
        int stateSize

# cdef extern from "Marmot/MarmotElementProperty.h":
#     cdef cppclass MarmotElementProperty nogil:
#         pass

#     cdef cppclass MarmotMaterialSection(MarmotElementProperty) nogil:
#         MarmotMaterialSection(int materialCode, const double* materialProperties, int nMaterialProperties)

# cdef extern from "Marmot/MarmotMeshfreeApproximation.h" namesp:
#     cdef cppclass MarmotMeshfreeApproximation nogil:
#         pass

# cdef extern from "Marmot/MarmotMeshfreeKernelFunction.h":
#     cdef cppclass MarmotMeshfreeKernelFunction nogil:
#         pass

cdef extern from "Marmot/MarmotParticleLibrary.h" namespace "MarmotLibrary" nogil:
    cdef cppclass MarmotParticleFactory:
        @staticmethod
        MarmotParticle* createParticle(const string& particleName,
                               int particleNumber,
                               const double* particleCoordinates,
                               int sizeParticleCoordinates,
                               double volume,
                               MarmotMaterialPoint& mp,
                               const MarmotMeshfreeApproximation& approximation) except +ValueError

cdef extern from "Marmot/MarmotParticle.h" namespace "Marmot::Meshfree":
    cdef cppclass MarmotParticle nogil:

        const vector[string]& getFields()

        const vector[int]& getDofIndicesPermutationPattern()

        void getVertexCoordinates(double* )

        void getVisualizationVertexCoordinates(double* )

        string getParticleShape()

        void assignMeshfreeKernelFunctions (const vector[const MarmotMeshfreeKernelFunction*] meshfreeKernelFunctions)

        void computePhysicsKernels(   const double* dUc,
                                            double* Pc,
                                            double* Kc,
                                            double timeNewTotal,
                                            double dT) except +

        void computeBodyLoad(               int type,
                                            const double* load,
                                            double* Pc,
                                            double* Kc,
                                            double timeNewTotal,
                                            double dT) except +

        void computeDistributedLoad(        int type,
                                            int surfaceID,
                                            const double* load,
                                            double* Pc,
                                            double* Kc,
                                            double timeNewTotal,
                                            double dT) except +

        const unordered_map[string, int]& getSupportedBodyLoadTypes()

        const unordered_map[string, int]& getSupportedDistributedLoadTypes()

        void getInterpolationVector( double* N, const double* coordinates)

        int getNumberOfRequiredStateVars()

        void assignStateVars( double* stateVars, int nStateVars )

        void initializeYourself()

        void acceptStateAndPosition()

        StateView getStateView( const string& stateName )

        int getDimension()

        int getNumberOfVertices()


cdef class MarmotParticleWrapper:

    cdef MarmotParticle* _marmotParticle
    cdef MarmotMaterialPoint* _marmotMaterialPoint
    cdef MarmotMeshfreeApproximation* _marmotMeshfreeApproximation

    cpdef void computePhysicsKernels(self, double[::1] dUc, double[::1] Rhs, double[::1] AMatrix, double timeNew, double dTime, ) nogil

    cdef int _number,
    cdef str _particleType,
    cdef str _ensightType
    cdef int _nVertices
    cdef list _fields

    cdef np.ndarray _dofIndicesPermutation

    cdef public double[::1] _stateVars
    cdef public double[::1] _stateVarsTemp
    cdef int _nStateVars
    cdef int _nDim

    cdef _assignedKernelFunctions

    cdef dict _supportedBodyLoads

    cdef dict _supportedDistributedLoads

    cdef public list _assignedShapeFunctions

    cdef public double[::1]  _materialProperties

    # cdef public np.ndarray _centerCoordinates
    cdef public np.ndarray _vertexCoordinates

    # cdef double[::1] _centerCoordinatesView
    cdef double[:,::1] _vertexCoordinatesView

    # nogil methods are already declared here

    cpdef void _initializeStateVarsTemp(self, ) nogil

    # cpdef void computeYourself(self,
    #                  double timeNew,
    #                  double dTime, ) except * nogil

    cdef double[::1] getStateView(self, string stateName)
