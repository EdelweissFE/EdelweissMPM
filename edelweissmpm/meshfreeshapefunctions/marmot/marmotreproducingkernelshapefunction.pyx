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


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotReprocingKernelShapeFunctionWrapper:
    # cdef classes cannot subclass.

    def __cinit__(self, Node node, int shapefunctionOrder, double supportRadius):
        """This C-level method is responsible for actually creating the MarmotMaterialPoint.

        Parameters
        ----------

            """
        self._node = node
        self._center = self._node.coordinates
        self._dimension = self._center.shape[0]
        self._shapefunctionOrder = shapefunctionOrder
        self._supportRadius = supportRadius
        self._marmotReproducingKernelShapeFunction = new MarmotReproducingKernelShapeFunction( self._center,
                                                                                              self._dimension,
                                                                                              self._supportRadius,
                                                                                              self._continuityOrder,
                                                                                              self._completenessOrder)

    def __dealloc__(self):
        del self._marmotReproducingKernelShapeFunction

    @property
    def node(self) -> Node:
        return self._node

    def getBoundingBox(self, ):

        cdef np.ndarray boundingBoxMin = np.zeros(self._dimension)
        cdef np.ndarray boundingBoxMax = np.zeros(self._dimension)
        cdef double[::1] boundingBoxMinView = boundingBoxMin
        cdef double[::1] boundingBoxMaxView = boundingBoxMax

        self._marmotReproducingKernelShapeFunction.getBoundingBox(&boundingBoxMinView[0], &boundingBoxMaxView[0])

        return boundingBoxMin, boundingBoxMax

    def isCoordinateInCurrentSupport(self, double[::1] coords) -> bool:
        return self._marmotReproducingKernelShapeFunction.isInSupport(&coords[0])

