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

from abc import ABC, abstractmethod
import numpy as np
from edelweissfe.points.node import Node


class MaterialPointBase(ABC):
    """A basic material point.

    Attributes
    ----------
    label : int
        The unique label (ID) of this material point.
    assignedCells : set

    Parameters
    ----------
    formulation : str
        The string describing the formulation of the material point.
    number : int
        The unique number (ID) of this material point.
    coordinates : np.ndarray
        The coordinates of the material point.
    volume : float
        The volume of the material point.
    """

    @abstractmethod
    def __init__(self, formulation: str, number: int, coordinates: np.ndarray, volume: float, material):
        pass

    @property
    @abstractmethod
    def number(self) -> int:
        """The unique label (ID) of this material point."""
        pass

    @property
    def assignedCells(self) -> set:
        """The list of currently assigned cells."""
        pass

    @abstractmethod
    def assignCells(self, cells: set):
        """Assign the list of cells in which the material point is currently residing."""
        pass

    @property
    @abstractmethod
    def ensightType(self) -> str:
        """The shape of the materialpoint in Ensight Gold notation."""
        pass

    @abstractmethod
    def getVertexCoordinates(
        self,
    ) -> np.ndarray:
        """A material point's shape is defined by an arbitrary number of vertices.

        Returns
        -------
        np.ndarray
            All coordinates for all bounding vertices."""
        pass

    @abstractmethod
    def getCenterCoordinates(
        self,
    ) -> np.ndarray:
        """The location of the material point is defined by the coordinates of it's center.

        Returns
        -------
        np.ndarray
            The coordinates of the material point's center of mass."""
        pass

    @abstractmethod
    def getVolume(
        self,
    ) -> float:
        """A material point has distinct volume, which may change during a simulation.

        Returns
        -------
        float
            The current volume occupied by this material point."""
        pass

    @abstractmethod
    def computeYourself(self, timeTotal: float, dT: float):
        """Compute the current material response."""

        pass

    @abstractmethod
    def acceptStateAndPosition(
        self,
    ):
        """Accept the computed state (in nonlinear iteration schemes) and the position."""

        pass

    @abstractmethod
    def prepareTimestep(self, timeTotal: float, dT: float):
        """Prepare a new time step, i.e., before interpolation from the grid takes place."""

        pass

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

        pass

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

        pass

    @abstractmethod
    def initializeYourself(
        self,
    ):
        """Initalize the mateiral point to be ready for computing."""
        pass

    @abstractmethod
    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        """Assign a material and material properties.

        Parameters
        ----------
        materialName
            The name of the requested material.
        materialProperties
            The numpy array containing the material properties.
        """
        pass

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

        pass
