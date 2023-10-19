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
Implementing your own cells can be done easily by subclassing from 
the abstract base class :class:`~BaseCell`.
"""

from abc import ABC, abstractmethod
import numpy as np
from fe.points.node import Node

from mpm.materialpoints.base.mp import BaseMaterialPoint


class BaseCell(ABC):
    @abstractmethod
    def __init__(self, cellType: str, cellNumber: int):
        """MPM cells in EdelweissFE should be derived from this
        base class in order to follow the general interface.

        EdelweissFE expects the layout of the internal and external load vectors, P, PExt, (and the stiffness)
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
        """

        pass

    @property
    @abstractmethod
    def cellNumber(self) -> int:
        """The unique number of this cell"""

        pass

    @property
    @abstractmethod
    def nNodes(self) -> int:
        """The list of nodes this cell holds"""

        pass

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The total number of degrees of freedom this cell has"""

        pass

    @property
    @abstractmethod
    def fields(self) -> list[list[str]]:
        """The list of fields per grid nodes."""

        pass

    @property
    @abstractmethod
    def dofIndicesPermutation(self) -> np.ndarray:
        """The permutation pattern for the residual vector and the stiffness matrix to
        aggregate all entries in order to resemble the defined fields nodewise."""

        pass

    @property
    @abstractmethod
    def ensightType(self) -> str:
        """The shape of the element in Ensight Gold notation."""
        pass

    @abstractmethod
    def setNodes(self, nodes: list[Node]):
        """Assign the nodes to the cell.

        Parameters
        ----------
        nodes
            A list of nodes.
        """
        pass

    @abstractmethod
    def interpolateSolutionContributionToMaterialPoints(
        self,
        materialPoints: list[BaseMaterialPoint],
        dU: np.ndarray,
    ):
        """Interpolate field solutions to MaterialPoints.

        Parameters
        ----------
        materialPoints
            The list of MaterialPoints for which the solution should be added.
        dU
            The current solution vector contribution increment for all fields.

        """

        pass

    @abstractmethod
    def computeMaterialPointKernels(
        self,
        materialPoints: list[BaseMaterialPoint],
        P: np.ndarray,
        K: np.ndarray,
        timeStep: float,
        timeTotal: float,
        dTime: float,
    ):
        """Evaluate the kernel residual and stiffness for given time based for all assigned MaterialPoints.

        Parameters
        ----------
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        U
            The current solution vector.
        dU
            The current solution vector increment.
        time
            Array of step time and total time.
        dTime
            The time increment.
        """

        pass

    @abstractmethod
    def getCoordinatesAtCenter(self) -> np.ndarray:
        """Compute the underlying MarmotElement centroid coordinates.

        Returns
        -------
        np.ndarray
            The cells's central coordinates.
        """

        pass

    @abstractmethod
    def isCoordinateInCell(self, coordinate: np.ndarray) -> bool:
        """Check if a given coordinate is located within this cell.

        Returns
        -------
        bool
            The truth value if this coordinate is located in the cell.
        """
        pass
