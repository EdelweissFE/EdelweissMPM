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
Cells represent the basic building blocks of the MPM grid. They are used to
compute the material point kernels and to interpolate the field solutions back
to the material points. Cells are used to compute the residual and stiffness
matrices for the global system of equations.

Implementing your own cells can be done easily by subclassing from
the abstract base class :class:`~CellBase`.
"""

from abc import ABC, abstractmethod

import numpy as np
from edelweissfe.points.node import Node

from edelweissmpm.materialpoints.base.mp import MaterialPointBase


class CellBase(ABC):
    @abstractmethod
    def __init__(self, cellType: str, cellNumber: int, nodes: list[Node]):
        """MPM cells in EdelweissMPM should be derived from this
        base class in order to follow the general interface.

        EdelweissMPM expects the layout of the internal and external load vectors, P, PExt, (and the stiffness)
        to be of the form

        .. code-block:: console

            [ node 1 - dofs field 1,
              node 1 - dofs field 2,
              node 1 - ... ,
              node 1 - dofs field n,
              node 2 - dofs field 1,
              ... ,
              node N - dofs field n].

        Parameters
        ----------
        cellType
            A string identifying the requested element formulation.
        cellNumber
            A unique integer label used for all kinds of purposes.
        nodes
            The list of Nodes assigned to this cell.
        """

    @property
    @abstractmethod
    def cellNumber(self) -> int:
        """The unique number of this cell"""

    @property
    @abstractmethod
    def nNodes(self) -> int:
        """The list of nodes this cell holds"""

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The total number of degrees of freedom this cell has"""

    @property
    @abstractmethod
    def fields(self) -> list[list[str]]:
        """The list of fields per grid nodes."""

    @property
    @abstractmethod
    def dofIndicesPermutation(self) -> np.ndarray:
        """The permutation pattern for the residual vector and the stiffness matrix to
        aggregate all entries in order to resemble the defined fields nodewise."""

    @property
    @abstractmethod
    def ensightType(self) -> str:
        """The shape of the element in Ensight Gold notation."""

    @property
    @abstractmethod
    def assignedMaterialPoints(self) -> list[MaterialPointBase]:
        """The shape of the element in Ensight Gold notation."""

    @abstractmethod
    def assignMaterialPoints(self, materialPoints: list[MaterialPointBase]):
        """Assign a list of material points which are currently residing within the cell.

        Parameters
        ----------
        materialPoints
            The list of material points to be assigned.
        """

    @abstractmethod
    def interpolateSolutionContributionToMaterialPoints(
        self,
        dU: np.ndarray,
    ):
        """Interpolate field solutions to the assigned MaterialPoints.

        Parameters
        ----------
        dU
            The current solution vector contribution increment for all fields.

        """

    @abstractmethod
    def computeMaterialPointKernels(
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
        materialPoint: MaterialPointBase,
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
        materialPoint
            The specific (already assigned) mateiral point for computing the surface load.
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
    def getCoordinatesAtCenter(self) -> np.ndarray:
        """Compute the underlying MarmotElement centroid coordinates.

        Returns
        -------
        np.ndarray
            The cells's central coordinates.
        """

    @abstractmethod
    def isCoordinateInCell(self, coordinate: np.ndarray) -> bool:
        """Check if a given coordinate is located within this cell.

        Returns
        -------
        bool
            The truth value if this coordinate is located in the cell.
        """

    @abstractmethod
    def getBoundingBox(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounding box min. and max. of a cell

        Returns
        -------
        tuple
                The tuple containing the min. and max. coordinates of the bounding box.
        """

    @abstractmethod
    def getInterpolationVector(self, coordinate: np.ndarray) -> np.ndarray:
        """Get the interpolation vector for a given global coordinate.

        Returns
        -------
        np.ndarray
            The interpolation vector for all nodes.
        """
