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


cdef extern from "Marmot/MarmotMPMLibrary.h" namespace "MarmotLibrary" nogil:

    cdef cppclass MarmotCellFactory:
        @staticmethod
        MarmotCell* createCell(const string& cellName,
                               int cellNumber,
                               const double* nodeCoordinates,
                               int sizeNodeCoordinates) except +ValueError

        @staticmethod
        MarmotCell* createBSplineCell(const string& cellName,
                               int cellNumber,
                               const double* nodeCoordinates,
                               int sizeNodeCoordinates,
                               const double* knotVectors,
                               int sizeKnotVectors) except +ValueError

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

        void computeMaterialPointKernels(   const double* dUc,
                                            double* Pc,
                                            double* Kc,
                                            double timeNewTotal,
                                            double dT) except +

        void computeLumpedInertia( double* M ) except +

        void computeConsistentInertia( double* M ) except +

        void computeBodyLoad(               int type,
                                            const double* load,
                                            double* Pc,
                                            double* Kc,
                                            double timeNewTotal,
                                            double dT) except +

        void computeDistributedLoad(        int type,
                                            int surfaceID,
                                            int materialPointNumber,
                                            const double* load,
                                            double* Pc,
                                            double* Kc,
                                            double timeNewTotal,
                                            double dT) except +

        const unordered_map[string, int]& getSupportedBodyLoadTypes()

        const unordered_map[string, int]& getSupportedDistributedLoadTypes()

        void interpolateFieldsToMaterialPoints( const double* Q)

        void getInterpolationVector( double* N, const double* coordinates)

        void getBoundingBox(double* boundingBoxMin, double* boundingBoxMax)

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

    cdef dict _supportedDistributedLoads

    cdef double[:, ::1] _nodeCoordinates

    cdef list _assignedMaterialPoints

    cpdef void computeMaterialPointKernels(self, double[::1] dUc, double[::1] Rhs, double[::1] AMatrix, double timeNew, double dTime, ) nogil

    cpdef void computeLumpedInertia(self, double[::1] M ) nogil

    cpdef void computeConsistentInertia(self, double[::1] M ) nogil

    cpdef void interpolateFieldsToMaterialPoints(self,
                     double[::1] dUc ) nogil
