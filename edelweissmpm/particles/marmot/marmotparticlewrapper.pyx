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

cimport edelweissmpm.particles.marmot.marmotparticlewrapper

from edelweissfe.utils.exceptions import CutbackRequest

from cython.operator cimport dereference
from libc.stdlib cimport free, malloc
from libcpp.memory cimport allocator, make_unique, unique_ptr

from edelweissmpm.materialpoints.marmotmaterialpoint.mp cimport (
    MarmotMaterialPoint,
    MarmotMaterialPointWrapper,
)
from edelweissmpm.meshfree.approximations.marmot.marmotmeshfreeapproximation cimport (
    MarmotMeshfreeApproximation,
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction cimport (
    MarmotMeshfreeKernelFunction,
    MarmotMeshfreeKernelFunctionWrapper,
)


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotParticleWrapper:
    """This cell as a wrapper for MarmotCells.
    It is an abstract class and cannot be used directly.
    Rather, it is used as a base class for the specific cell types, such as LagrangianMarmotCell and BSplineMarmotCell.

    For the documentation of MarmotCells, please refer to `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.

    Parameters
    ----------
    cellType
        The Marmot element which should be represented, e.g., CPE4.
    cellNumber
        The (unique) number of this cell.
    nodes
        The list of nodes for this Cell.
        """


    def __cinit__(self,
                  str particleType,
                  int particleNumber,
                  np.ndarray vertexCoordinates,
                  double volume,
                  MarmotMaterialPointWrapper marmotMaterialPointWrapper,
                  MarmotMeshfreeApproximationWrapper marmotMeshfreeApproximationWrapper,
                  ):

        self._vertexCoordinates = np.copy(vertexCoordinates)
        self._vertexCoordinatesView = self._vertexCoordinates

        self._marmotMaterialPoint = <MarmotMaterialPoint*> marmotMaterialPointWrapper._marmotMaterialPoint
        self._marmotMeshfreeApproximation = <MarmotMeshfreeApproximation* > marmotMeshfreeApproximationWrapper._marmotMeshfreeApproximation

        self._assignedKernelFunctions = list()

        try:
            self._marmotParticle = MarmotParticleFactory.createParticle(particleType.encode('utf-8'),
                                                                                       particleNumber,
                                                                                       &self._vertexCoordinatesView[0,0],
                                                                                       self._vertexCoordinates.size,
                                                                                       volume,
                                                                                       self._marmotMaterialPoint[0],
                                                                                       self._marmotMeshfreeApproximation[0]
                                                                                       )
        except IndexError:
            raise NotImplementedError("Failed to create instance of MarmotParticle {:}.".format(particleType))

        self._nStateVars =           self._marmotParticle.getNumberOfRequiredStateVars()

        self._stateVars =            np.zeros(self._nStateVars)
        self._stateVarsTemp =        np.zeros(self._nStateVars)

        self._marmotParticle.assignStateVars(&self._stateVarsTemp[0], self._nStateVars)

        # self._cellNumber = cellNumber
        self._particleType = particleType
        self._number                = particleNumber
        self._ensightType           = self._marmotParticle.getParticleShape().decode('utf-8')
        self._nVertices             = self._marmotParticle.getNumberOfVertices()
        self._nDim                  = self._marmotParticle.getDimension()
        self._assignedShapeFunctions = list()

        # self._centerCoordinates = np.ndarray(self._nDim)
        # self._centerCoordinatesView = self._centerCoordinates

        cdef vector[string] fields           = self._marmotParticle.getFields()
        self._fields                         = [  s.decode('utf-8')  for s in fields ]

        cdef vector[int] permutationPattern = self._marmotParticle.getDofIndicesPermutationPattern()
        self._dofIndicesPermutation          = np.asarray(permutationPattern)

        cdef dict supportedBodyLoads = self._marmotParticle.getSupportedBodyLoadTypes()
        self._supportedBodyLoads = {k.decode() :  v for k, v in supportedBodyLoads.items()  }

        cdef dict supportedDistributedLoads = self._marmotParticle.getSupportedDistributedLoadTypes()
        self._supportedDistributedLoads = {k.decode() :  v for k, v in supportedDistributedLoads.items()  }

        # cdef MarmotMeshfreeApproximationWrapper mfaWrapper = approximation

        # cdef const MarmotMeshfreeApproximation* mfa = <const MarmotMeshfreeApproximation*> (  mfaWrapper._marmotMeshfreeApproximation   )
        # self._marmotParticle.assignApproximationType( mfa[0] )

    # @property
    # def cellNumber(self):
    #     return self._cellNumber

    # @property
    # def cellType(self):
    #     return self._cellType

    @property
    def nodes(self):
        return [sf.node for sf in self._assignedShapeFunctions ]

    @property
    def nNodes(self):
        return len(self._shapeFunctions)

    # @property
    # def nDof(self):
    #     return self._nDof

    @property
    def fields(self):
        return self._fields

    @property
    def dofIndicesPermutation(self):
        return self._dofIndicesPermutation

    @property
    def ensightType(self):
        return self._ensightType

    # @property
    # def assignedMaterialPoints(self):
    #     return self._assignedMaterialPoints

    @property
    def number(self):
        return self._number

    @property
    def materialPointType(self):
        return self._materialPointType

    cpdef void computePhysicsKernels(self,
                         double[::1] dUc,
                         double[::1] Pc,
                         double[::1] Kc,
                         double timeNew,
                         double dTime, ) nogil:
        """Evaluate residual and stiffness for given time, field, and field increment."""

        self._marmotParticle.computePhysicsKernels(&dUc[0], &Pc[0], &Kc[0], timeNew, dTime)

    # cpdef void interpolateFieldsToMaterialPoints(self, double[::1] dUc) nogil:

    #     self._marmotCell.interpolateFieldsToMaterialPoints(&dUc[0])

    def computeBodyLoad(self,
                         str loadType,
                         double[::1] load,
                         double[::1] Pc,
                         double[::1] Kc,
                         double timeNew,
                         double dTime):

        self._marmotParticle.computeBodyLoad( self._supportedBodyLoads[loadType.upper()], &load[0], &Pc[0], &Kc[0], timeNew, dTime)

    def computeDistributedLoad(self,
                         str loadType,
                         int surfaceID,
                         # materialPoint,
                         double[::1] load,
                         double[::1] Pc,
                         double[::1] Kc,
                         double timeNew,
                         double dTime):

        pass
        # cdef int mpNumber = materialPoint.number

        # self._marmotParticle.computeDistributedLoad( self._supportedDistributedLoads[loadType.upper()],
        #                                         surfaceID,
        #                                         &load[0],
        #                                         &Pc[0],
        #                                         &Kc[0],
        #                                         timeNew,
        #                                         dTime)

    def getInterpolationVector(self, coordinate: np.ndarray) -> np.ndarray:
        pass
        # cdef double[::1] coords = coordinate
        # cdef np.ndarray N = np.zeros(self._nNodes)
        # cdef double[::1] Nview_ = N
        # self._marmotCell.getInterpolationVector(&Nview_[0], &coords[0])
        # return N


    def assignKernelFunctions(self, list marmotMeshfreeKernelFunctionWrappers):
        self._assignedKernelFunctions = marmotMeshfreeKernelFunctionWrappers

    def getAssignedKernelFunctions(self):
        return self._assignedKernelFunctions

        # cdef vector[MarmotMeshfreeShapeFunction*] sfs
        # cdef MarmotMeshfreeApproximationWrapper sfWrapper

        # for sfWrapper in marmotMeshfreeShapeFunctionWrappers:
        #     sfs.push_back(<MarmotMeshfreeShapeFunction*> sfWrapper._marmotMeshfreeShapeFunction)

        # self._marmotParticle.assignShapeFunctions(sfs)

        # self._assignedMaterialPoints = validMarotMeshfreeShapeFunctions


    def acceptStateAndPosition(self,):
        self._marmotMaterialPoint.acceptStateAndPosition()
        self._stateVars[:] = self._stateVarsTemp

    def initializeYourself(self):
        self._stateVarsTemp[:] = self._stateVars
        self._marmotMaterialPoint.initializeYourself()
        self.acceptStateAndPosition()

    # def prepareYourself(self, timeTotal: float, dTime: float):
    #     self._stateVarsTemp[:] = self._stateVars
    #     self._marmotMaterialPoint.prepareYourself(timeTotal, dTime)

    cpdef void _initializeStateVarsTemp(self, ) nogil:
        self._stateVarsTemp[:] = self._stateVars

    def getResultArray(self, result:str, getPersistentView:bool=True):
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by the underlying MarmotMaterialPoint."""

        cdef string result_ =  result.encode('UTF-8')
        return np.array(  self.getStateView(result_ ), copy= not getPersistentView)

    cdef double[::1] getStateView(self, string result ):
        """Directly access the state vars of the underlying MarmotElement"""

        cdef StateView res = self._marmotParticle.getStateView(result)

        return <double[:res.stateSize]> ( res.stateLocation )

    def getVertexCoordinates(self):
        """Get the underlying MarmotMaterialPoint vertex coordinates."""

        self._marmotParticle.getVertexCoordinates(&self._vertexCoordinatesView[0,0])

        return self._vertexCoordinates


    # def getCenterCoordinates(self):
    #     """Compute the underlying MarmotMaterialPoint center of mass coordinates."""
    #     self._marmotMaterialPoint.getVertexCoordinates(&self._centerCoordinatesView[0])

    #     return self._centerCoordinates

    # cpdef void computeYourself(self,
    #                      double timeNew,
    #                      double dTime, ) except * nogil:
    #     """Evaluate residual and stiffness for given time, field, and field increment."""

    #     self._marmotMaterialPoint.computeYourself(timeNew, dTime)

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        pass

    # def _assignMaterial(self, materialName: str, materialProperties: np.ndarray):
    #     """Assign a material and material properties to the underlying MarmotElement.
    #     Furthermore, create two sets of state vars:

    #         - the actual set,
    #         - and a temporary set for backup in nonlinear iteration schemes.
    #     """

    #     self._materialProperties =  materialProperties

    #     try:
    #         self._marmotParticle.assignMaterial(
    #                 MarmotMaterialSection(
    #                         MarmotMaterialFactory.getMaterialCodeFromName(
    #                                 materialName.upper().encode('UTF-8')),
    #                         &self._materialProperties[0],
    #                         self._materialProperties.shape[0] ) )
    #     except IndexError:
    #         raise NotImplementedError("Marmot material {:} not found in library.".format(materialName))


    def __dealloc__(self):
        del self._marmotParticle
