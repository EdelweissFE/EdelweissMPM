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

cimport edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotMeshfreeKernelFunctionWrapper:
    # cdef classes cannot subclass.
    def __init__(self, node, str kernelType, *args, **kwargs):
        pass

    def __cinit__(self, node, str kernelType, *args, **kwargs):

        self._node = node
        self._center = np.copy(self._node.coordinates)
        cdef double[::1] center = self._center
        self._dimension = self._center.shape[0]

        self._boundingBoxMin = np.zeros(self._dimension)
        self._boundingBoxMax = np.zeros(self._dimension)

        if kernelType == "BSplineBoxed":
            # cdef double supportRadius = kwargs.get("supportRadius", 1.0)
            # cdef int continuityOrder = kwargs.get("continuityOrder", 2)

            self._marmotMeshfreeKernelFunction = <MarmotMeshfreeKernelFunction*> (new MarmotMeshfreeKernelFunctionBSplineBoxed( &center[0], self._dimension,
                                                                                              kwargs.get("supportRadius", 1.0),
                                                                                              kwargs.get("continuityOrder", 2)
                                                                                              ))
        else:
            raise ValueError("Unknown kernel type {:s}, supported types are 'BSplineBoxed'".format(kernelType))

    def __dealloc__(self):
        del self._marmotMeshfreeKernelFunction

    @property
    def node(self) -> Node:
        return self._node

    @property
    def center(self) -> np.ndarray:
        return self._center

    def move(self, double[::1] displacement):
        # self._displacement = np.copy(displacement)
        # cdef double[::1] displacementView = self._displacement
        self._marmotMeshfreeKernelFunction.move(&displacement[0])

    def getBoundingBox(self, ):

        cdef double[::1] boundingBoxMinView = self._boundingBoxMin
        cdef double[::1] boundingBoxMaxView = self._boundingBoxMax

        self._marmotMeshfreeKernelFunction.getBoundingBox(&boundingBoxMinView[0], &boundingBoxMaxView[0])

        return self._boundingBoxMin, self._boundingBoxMax

    def isCoordinateInCurrentSupport(self, double[::1] coords) -> bool:
        cdef int isInside = self._marmotMeshfreeKernelFunction.isInSupport(&coords[0])
        return isInside

    def computeKernelFunction(self, double[::1] coords) -> double:
        return self._marmotMeshfreeKernelFunction.computeKernelFunction(&coords[0])
