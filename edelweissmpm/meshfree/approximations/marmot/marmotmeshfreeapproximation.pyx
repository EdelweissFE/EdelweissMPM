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

cimport edelweissmpm.meshfree.approximations.marmot.marmotmeshfreeapproximation
from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction cimport (
    MarmotMeshfreeKernelFunction,
    MarmotMeshfreeKernelFunctionWrapper,
)


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotMeshfreeApproximationWrapper:

    def __cinit__(self, str approximationType, int dim, **kwargs):

        validApproximationTypes = ["ReproducingKernel, ReproducingKernelImplicitGradient"]

        self._nDim = dim

        if approximationType == "ReproducingKernel":
                self._marmotMeshfreeApproximation = <MarmotMeshfreeApproximation*> (
                                                                                    new
                                                                                    MarmotMeshfreeReproducingKernelApproximation( dim, kwargs['completenessOrder'] ) )
        elif approximationType == "ReproducingKernelImplicitGradient":
                self._marmotMeshfreeApproximation = <MarmotMeshfreeApproximation*> (
                                                                                    new
                                                                                    MarmotMeshfreeReproducingKernelApproximationImplicit( dim, kwargs['completenessOrder'] ) )
        else:
                raise NotImplementedError("Approximation type {:} not found in library. Valid options are {:}".format(approximationType, validApproximationTypes))

    def computeShapeFunctions(self, double[::1] coordinates, list marmotMeshfreeKernelFunctionWrappers):
        cdef vector[const MarmotMeshfreeKernelFunction*] mMFKFs
        cdef MarmotMeshfreeKernelFunctionWrapper mMFKFWrapper

        for mMFKFWrapper in marmotMeshfreeKernelFunctionWrappers:
            mMFKFs.push_back(<MarmotMeshfreeKernelFunction*> mMFKFWrapper._marmotMeshfreeKernelFunction)

        cdef result = np.empty(len(marmotMeshfreeKernelFunctionWrappers), dtype=np.float64)
        cdef double[::1] result_view = result
        self._marmotMeshfreeApproximation.computeShapeFunctions(&coordinates[0], mMFKFs, &result_view[0])

        return result

    def computeShapeFunctionsAndGradients(self, double[::1] coordinates, list marmotMeshfreeKernelFunctionWrappers):
        cdef vector[const MarmotMeshfreeKernelFunction*] mMFKFs
        cdef MarmotMeshfreeKernelFunctionWrapper mMFKFWrapper

        for mMFKFWrapper in marmotMeshfreeKernelFunctionWrappers:
            mMFKFs.push_back(<MarmotMeshfreeKernelFunction*> mMFKFWrapper._marmotMeshfreeKernelFunction)

        cdef result = np.empty(len(marmotMeshfreeKernelFunctionWrappers), dtype=np.float64)
        cdef resultGradient = np.empty( (self._nDim, len(marmotMeshfreeKernelFunctionWrappers) ), dtype=np.float64, order='F')
        cdef double[::1] result_view = result
        cdef double[::1, :] resultGradient_view = resultGradient
        self._marmotMeshfreeApproximation.computeShapeFunctionsAndGradients(&coordinates[0], mMFKFs, &result_view[0], &resultGradient_view[0, 0])

        return result, resultGradient.T

    def __dealloc__(self):
        del self._marmotMeshfreeApproximation

