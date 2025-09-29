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

cimport edelweissmpm.materialpoints.marmotmaterialpoint.mp

from edelweissfe.utils.exceptions import CutbackRequest

from libc.stdlib cimport free, malloc
from libcpp.memory cimport allocator, make_unique, unique_ptr


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotMaterialPointWrapper:
    # cdef classes cannot subclass. Hence we do not subclass from the BaseMaterialPoint,
    # but still we follow the interface for compatiblity.

    def __cinit__(self, materialPointType,
                  int materialPointNumber,
                  np.ndarray vertexCoordinates,
                  double volume,
                  material
                  ):
        """This C-level method is responsible for actually creating the MarmotMaterialPoint.

        Parameters
        ----------
        materialPointType
            The MarmotMaterialPoint which should be represented.
        materialPointNumber
            The unique number of the MaterialPoint."""


        self._vertexCoordinates = np.copy(vertexCoordinates)
        self._vertexCoordinatesView = self._vertexCoordinates

        try:
            self._marmotMaterialPoint = MarmotMaterialPointFactory.createMaterialPoint(materialPointType.encode('utf-8'),
                                                                                       materialPointNumber,
                                                                                       &self._vertexCoordinatesView[0,0],
                                                                                       self._vertexCoordinates.size,
                                                                                       volume
                                                                                       )
        except ValueError:
            raise NotImplementedError("Failed to create instance of MarmotMaterialPoint {:}.".format(materialPointType))

        self._number                = materialPointNumber
        self._materialPointType     = materialPointType
        self._ensightType           = self._marmotMaterialPoint.getMaterialPointShape().decode('utf-8')
        self._nVertices             = self._marmotMaterialPoint.getNumberOfVertices()
        self._nDim                  = self._marmotMaterialPoint.getDimension()
        self._assignedCells         = list()

        self._centerCoordinates = np.ndarray(self._nDim)
        self._centerCoordinatesView = self._centerCoordinates

        cdef str materialName = material['material']
        cdef np.ndarray materialProperties = material['properties']

        self._assignMaterial(materialName, materialProperties)


    @property
    def number(self):
        return self._number

    @property
    def assignedCells(self) -> set:
        """The list of currently attached cells."""
        return self._assignedCells

    def assignCells(self, list cells):
        """The list of currently attached cells."""
        self._assignedCells = cells

    @property
    def materialPointType(self):
        return self._materialPointType

    @property
    def ensightType(self):
        return self._ensightType

    def acceptStateAndPosition(self,):
        self._marmotMaterialPoint.acceptStateAndPosition()
        self._stateVars[:] = self._stateVarsTemp

    def initializeYourself(self):
        self._stateVarsTemp[:] = self._stateVars
        self._marmotMaterialPoint.initializeYourself()
        self.acceptStateAndPosition()

    def prepareYourself(self, timeTotal: float, dTime: float):
        self._stateVarsTemp[:] = self._stateVars
        self._marmotMaterialPoint.prepareYourself(timeTotal, dTime)

    cpdef void _initializeStateVarsTemp(self, ) nogil:
        self._stateVarsTemp[:] = self._stateVars

    def getResultArray(self, result:str, getPersistentView:bool=True):
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by the underlying MarmotMaterialPoint."""

        cdef string result_ =  result.encode('UTF-8')
        return np.array(  self.getStateView(result_ ), copy= not getPersistentView)

    cdef double[::1] getStateView(self, string result ):
        """Directly access the state vars of the underlying MarmotElement"""

        cdef StateView res = self._marmotMaterialPoint.getStateView(result)

        return <double[:res.stateSize]> ( res.stateLocation )

    def getVertexCoordinates(self):
        """Get the underlying MarmotMaterialPoint vertex coordinates."""

        self._marmotMaterialPoint.getVertexCoordinates(&self._vertexCoordinatesView[0,0])

        return self._vertexCoordinates


    def getCenterCoordinates(self):
        """Compute the underlying MarmotMaterialPoint center of mass coordinates."""
        self._marmotMaterialPoint.getVertexCoordinates(&self._centerCoordinatesView[0])

        return self._centerCoordinates

    cpdef void computeYourself(self,
                         double timeNew,
                         double dTime, ) except * nogil:
        """Evaluate residual and stiffness for given time, field, and field increment."""

        self._marmotMaterialPoint.computeYourself(timeNew, dTime)

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        pass

    def _assignMaterial(self, materialName: str, materialProperties: np.ndarray):
        """Assign a material and material properties to the underlying MarmotElement.
        Furthermore, create two sets of state vars:

            - the actual set,
            - and a temporary set for backup in nonlinear iteration schemes.
        """

        self._materialProperties =  materialProperties

        try:
            self._marmotMaterialPoint.assignMaterial(
                    MarmotMaterialSection(
                            MarmotMaterialFactory.getMaterialCodeFromName(
                                    materialName.upper().encode('UTF-8')),
                            &self._materialProperties[0],
                            self._materialProperties.shape[0] ) )
        except IndexError:
            raise NotImplementedError("Marmot material {:} not found in library.".format(materialName))

        self._nStateVars =           self._marmotMaterialPoint.getNumberOfRequiredStateVars()

        self._stateVars =            np.zeros(self._nStateVars)
        self._stateVarsTemp =        np.zeros(self._nStateVars)

        self._marmotMaterialPoint.assignStateVars(&self._stateVarsTemp[0], self._nStateVars)

    def __dealloc__(self):
        del self._marmotMaterialPoint
