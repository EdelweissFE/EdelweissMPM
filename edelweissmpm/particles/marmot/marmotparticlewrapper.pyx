#Initalize  -*- coding: utf-8 -*-
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
    """This class as a wrapper for MarmotParticles.

    For the documentation of MarmotParticles, please refer to `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.
    """


    def __cinit__(self,
                  str particleType,
                  int particleNumber,
                  double[:,::1] vertexCoordinates,
                  double volume,
                  MarmotMeshfreeApproximationWrapper marmotMeshfreeApproximationWrapper,
                  dict material
                  ):


        ##TODO: This this crap:
        # mpCenter =  np.mean( vertexCoordinates, axis=0).reshape(-1, vertexCoordinates.shape[1])

        # self._mp = MarmotMaterialPointWrapper("GradientEnhancedMicropolar/PlaneStrain", particleNumber, mpCenter , volume, material)

        self._vertexCoordinates = np.copy(vertexCoordinates)
        self._vertexCoordinatesView = self._vertexCoordinates





        self._centerCoordinates = np.zeros(vertexCoordinates.shape[1])
        self._centerCoordinatesView = self._centerCoordinates

        # self._marmotMaterialPoint = <MarmotMaterialPoint*> self._mp._marmotMaterialPoint
        self._marmotMeshfreeApproximation = <MarmotMeshfreeApproximation* > marmotMeshfreeApproximationWrapper._marmotMeshfreeApproximation

        self._assignedKernelFunctions = list()

        cdef str materialName = material['material']
        self.materialProperties = material['properties']
        self.materialPropertiesView = self.materialProperties
        cdef int nMaterialProperties = len(self.materialProperties)

        try:
            self._marmotParticle = MarmotParticleFactory.createParticle(particleType.encode('utf-8'),
                                                                                       particleNumber,
                                                                                       &self._vertexCoordinatesView[0,0],
                                                                                       self._vertexCoordinates.size,
                                                                                       volume,
                                                                                       materialName.upper().encode('utf-8'),
                                                                                       &self.materialPropertiesView[0],
                                                                                       nMaterialProperties,
                                                                                       self._marmotMeshfreeApproximation[0],
                                                                                       )
        except IndexError:
            raise NotImplementedError("Failed to create instance of MarmotParticle {:}.".format(particleType))

        self._nStateVars =           self._marmotParticle.getNumberOfRequiredStateVars()

        self._stateVars =            np.zeros(self._nStateVars)
        self._stateVarsTemp =        np.zeros(self._nStateVars)

        self._marmotParticle.assignStateVars(&self._stateVarsTemp[0], self._nStateVars)

        self._particleType          = particleType
        self._number                = particleNumber
        self._ensightType           = self._marmotParticle.getParticleShape().decode('utf-8')
        self._nVertices             = self._marmotParticle.getNumberOfVertices()
        self._nDim                  = self._marmotParticle.getDimension()
        self._nBaseDof              = self._marmotParticle.getNBaseDof()
        # self._assignedKernelFunctions = list()

        cdef vector[string] fields           = self._marmotParticle.getFields()
        self._baseFields                         = [  s.decode('utf-8')  for s in fields ]

        cdef dict supportedBodyLoads = self._marmotParticle.getSupportedBodyLoadTypes()
        self._supportedBodyLoads = {k.decode() :  v for k, v in supportedBodyLoads.items()  }

        cdef dict supportedDistributedLoads = self._marmotParticle.getSupportedDistributedLoadTypes()
        self._supportedDistributedLoads = {k.decode() :  v for k, v in supportedDistributedLoads.items()  }

        self._nEvaluationPoints = self._marmotParticle.getNumberOfEvaluationPoints()

        self._evaluationCoordinates = np.zeros( ( self._nEvaluationPoints, self._nDim) )
        self._evaluationCoordinatesView = self._evaluationCoordinates

    @property
    def nodes(self):
        return self._nodes

    @property
    def dimension(self):
        return self._nDim

    @property
    def baseFields(self):
        return self._baseFields

    @property
    def fields(self):
        return self._fields

    @property
    def propertyNames(self) -> list[str]:
        """Get the list of valid properties for this particle."""

        return self._marmotParticle.getPropertyNames()

    def setProperty(self, str propertyName, value):
        """Set a property of the particle."""

        cdef double[::1] _value = np.array(value, dtype=np.float64).reshape((1,))

        self._marmotParticle.setProperty(propertyName.encode('utf-8'), &_value[0])

    def setProperties(self, double[::1] properties):
        """Set a list of properties of the particle."""
        self._marmotParticle.setProperties(&properties[0], len(properties))


    @property
    def nDof(self):
        return self._nBaseDof * self._nAssignedKernelFunctions

    @property
    def ensightType(self):
        return self._ensightType

    @property
    def number(self):
        return self._number

    @property
    def kernelFunctions(self):
        return self._assignedKernelFunctions

    @property
    def dofIndicesPermutation(self):
        return None

    cpdef void computePhysicsKernels(self,
                                     double[::1] dUc,
                                     double[::1] Pc,
                                     double[::1] Kc,
                                     double timeNew,
                                     double dTime, ) nogil:
        """Evaluate residual and stiffness for given time, field, and field increment."""

        self._initializeStateVarsTemp()
        self._marmotParticle.computePhysicsKernels(&dUc[0], &Pc[0], &Kc[0], timeNew, dTime)

    cpdef void computeLumpedInertia( self, double[::1] M ) nogil:
        self._marmotParticle.computeLumpedInertia(&M[0])

    cpdef void computeConsistentInertia( self, double[::1] M ) nogil:
        self._marmotParticle.computeConsistentInertia(&M[0])

    cpdef void _initializeStateVarsTemp(self, ) nogil:
        self._stateVarsTemp[:] = self._stateVars

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
                               double[::1] load,
                               double[::1] Pc,
                               double[::1] Kc,
                               double timeNew,
                               double dTime):

        self._marmotParticle.computeDistributedLoad( self._supportedDistributedLoads[loadType.upper()], surfaceID, &load[0], &Pc[0], &Kc[0], timeNew, dTime)

    def getInterpolationVector(self, double[::1] coordinates) -> np.ndarray:
        cdef np.ndarray N = np.zeros(len(self._nodes))
        cdef double[::1] Nview_ = N
        self._marmotParticle.getInterpolationVector(&Nview_[0], &coordinates[0])
        return N

    def assignKernelFunctions(self, list marmotMeshfreeKernelFunctionWrappers):
        self._assignedKernelFunctions = marmotMeshfreeKernelFunctionWrappers
        self._nAssignedKernelFunctions = len(marmotMeshfreeKernelFunctionWrappers)

        cdef vector[const MarmotMeshfreeKernelFunction*] kernelFunctions

        cdef MarmotMeshfreeKernelFunctionWrapper k
        for k in marmotMeshfreeKernelFunctionWrappers:
            kernelFunctions.push_back(k._marmotMeshfreeKernelFunction)

        self._marmotParticle.assignMeshfreeKernelFunctions(kernelFunctions)

        self._nodes = [kf.node for kf in self._assignedKernelFunctions]
        self._fields = [ self._baseFields for n in self._nodes ]

    def prepareYourself(self, timeTotal: float, dTime: float):
        self._stateVarsTemp[:] = self._stateVars
        self._marmotParticle.acceptStateAndPosition() # this is to set the state of the particle to "ready"
        self._stateVarsTemp[:] = self._stateVars

    def acceptStateAndPosition(self,):
        self._marmotParticle.acceptStateAndPosition() # this is to set the state of the particle to "ready"
        self._stateVars[:] = self._stateVarsTemp

    def initializeYourself(self):
        self._stateVarsTemp[:] = self._stateVars
        self._marmotParticle.initializeYourself()
        self.acceptStateAndPosition()

    def getResultArray(self, result:str, getPersistentView:bool=True, qp:int=0):
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by the underlying MarmotParticle."""

        cdef string result_ =  result.encode('UTF-8')
        return np.array(  self.getStateView(result_, qp), copy= not getPersistentView, )

    cdef double[::1] getStateView(self, string result, int qp):
        """Directly access the state vars of the underlying MarmotElement"""

        cdef StateView res = self._marmotParticle.getStateView(result, qp)

        return <double[:res.stateSize]> ( res.stateLocation )

    def getVertexCoordinates(self):
        """Get the underlying MarmotParticle vertex coordinates."""

        self._marmotParticle.getVertexCoordinates(&self._vertexCoordinatesView[0,0])

        return self._vertexCoordinates

    def getEvaluationCoordinates(self):

        self._marmotParticle.getEvaluationCoordinates(&self._evaluationCoordinatesView[0,0])

        return self._evaluationCoordinates

    def getCenterCoordinates(self):
        """Get the underlying MarmotParticle center coordinates."""

        self._marmotParticle.getCenterCoordinates(&self._centerCoordinatesView[0])

        return self._centerCoordinates

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        pass


    def vci_getNumberOfConstraints(self, ):
        return self._marmotParticle.vci_getNumberOfConstraints()

    def vci_compute_Test_P_BoundaryIntegral(self, double[:,:,::1] r_AiC, double[::1] boundarySurfaceVector, int boundaryFaceID ):
        self._marmotParticle.vci_compute_Test_P_BoundaryIntegral(&r_AiC[0,0,0], &boundarySurfaceVector[0], boundaryFaceID )

    def vci_compute_TestGradient_P_Integral(self, double[:,:,::1]r_AiC):
        self._marmotParticle.vci_compute_TestGradient_P_Integral(&r_AiC[0,0,0])

    def vci_compute_Test_PGradient_Integral(self, double[:,:,::1] r_AiC):
        self._marmotParticle.vci_compute_Test_PGradient_Integral(&r_AiC[0,0,0])

    def vci_compute_MMatrix(self, double[:,:,::1] m_ACD):
        self._marmotParticle.vci_compute_MMatrix(&m_ACD[0,0,0])

    def vci_assignTestFunctionCorrectionTerms(self, double[:,:,::1] eta_AjC):
        self._marmotParticle.vci_assignTestFunctionCorrectionTerms(&eta_AjC[0,0,0])

    def getRestartData(self,):
        return self._stateVars

    def readRestartData(self, data):
        np.asarray(self._stateVars)[:] = data[:]
        self._initializeStateVarsTemp()

    def __dealloc__(self):
        del self._marmotParticle
