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
"""
Together with meshfree kernel functions (:class:`~BaseMeshfreeKernelFunction`), particles represent the basic building blocks of a particle based method. They
are used for computing the quadrature.

Implementing your own particles can be done easily by subclassing from
the abstract base class :class:`~BaseParticle`.
"""

from abc import abstractmethod

import numpy as np
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)

from edelweissmpm.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)


class BaseParticle(BaseNodeCouplingEntity):
    """The BaseParticle class is an abstract base class for all particles.

    If you want to implement a new particle, you have to inherit from this class.

    Particles communicate with field variables (:class:`~FieldVariable`) through meshfree kernel functions (:class:`~BaseMeshfreeKernelFunction`),
    which are attached to nodes.
    During a simulation, particles are used to compute the quadrature of the weak form of the governing equations.
    Accordingly, like elements and cells, they are responsible for computing the residual and stiffness matrix.

    Naturally, and in contrast to elements, cells and cell elements, particles have an identical set of fields on each attached node.
    Furthermore, due to the varying number attached nodes, no permutation is allowed.
    For particles generally a node-wise layout is assumed.

    For instance: [node_1_displacement, node_1_temperature, node_2_displacement, node_2_temperature, ...]
    """

    @property
    @abstractmethod
    def baseFields(self) -> list[str]:
        """Defines which fields are attached to the particle."""

    @property
    @abstractmethod
    def kernelFunctions(self) -> list[BaseMeshfreeKernelFunction]:
        """The kernel functions assigned to the particle."""

    @abstractmethod
    def getBoundingBox(
        self,
    ) -> np.ndarray:
        """Get the bounding box of the particle.

        Returns
        -------
        np.ndarray
            All coordinates for all bounding vertices."""

    @abstractmethod
    def getVertexCoordinates(
        self,
    ) -> np.ndarray:
        """The vertices defining the shape of the particle.

        Returns
        -------
        np.ndarray
            All coordinates for all bounding vertices."""

    @abstractmethod
    def getCenterCoordinates(
        self,
    ) -> np.ndarray:
        """Get the center coordinates of the particle.

        Returns
        -------
        np.ndarray
            The center coordinates."""

    @abstractmethod
    def acceptStateAndPosition(
        self,
    ):
        """Accept the computed state (in nonlinear iteration schemes) and the position."""

    @abstractmethod
    def getResultArray(self, result: str, getPersistentView: bool = True) -> np.ndarray:
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by a MaterialPoint.

        Parameters
        ----------
        result
            The name of the result.
        getPersistentView
            If true, the returned array should be continiously updated by the element.

        Returns
        -------
        np.ndarray
            The result.
        """

    @abstractmethod
    def setProperties(self, propertyName: str, elementProperties: np.ndarray):
        """Assign a set of properties to the element.

        Parameters
        ----------
        propertyName
            The name of the property to be set.
        elementProperties
            A numpy array containing the element properties.
        """

    @abstractmethod
    def initializeYourself(
        self,
    ):
        """Initalize the particle to be ready for computing."""

    @abstractmethod
    def setInitialCondition(self, stateType: str, values: np.ndarray):
        """Assign initial conditions.

        Parameters
        ----------
        stateType
            The type of initial state.
        values
            The numpy array describing the initial state.
        """

    @abstractmethod
    def computePhysicsKernels(
        self,
        dU: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Evaluate the kernel residual and stiffness for given time based for all assigned MaterialPoints.

        Parameters
        ----------
        dU
            The current solution increment.
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        timeTotal
            The current total time.
        dTime
            The time increment.
        """

    @abstractmethod
    def computeBodyLoad(
        self,
        loadType: str,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute bulk loads (body loads) for given time based for all assigned MaterialPoints.

        Parameters
        ----------
        loadType
            The type of load to be computed (e.g., 'bodyforce')
        load
            The float (vector) describing the load.
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        timeTotal
            The current total time.
        dTime
            The time increment.
        """

    @abstractmethod
    def computeDistributedLoad(
        self,
        loadType: str,
        surfaceID: int,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute distributed (surface) a for given time based for a specific assigned material point.

        Parameters
        ----------
        loadType
            The type of load to be computed (e.g., 'pressure')
        surfaceID
            The ID describing the surface of the material point.
        load
            The float (vector) describing the load.
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        timeTotal
            The current total time.
        dTime
            The time increment.
        """

    @abstractmethod
    def assignMeshfreeKernelFunctions(self, kernelFunctions: list[BaseMeshfreeKernelFunction]):
        """Assign the meshfree kernel functions to the particle.

        Parameters
        ----------
        shapeFunctions
            The meshfree shape functions.
        """

    @abstractmethod
    def getInterpolationVector(self, coordinate: np.ndarray) -> np.ndarray:
        """Get the interpolation vector for a given global coordinate.

        Returns
        -------
        np.ndarray
            The interpolation vector for all nodes.
        """
