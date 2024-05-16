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
"""CellElements are a mixture of classical finite elements and cells of the MPM:
They work (and deform) like Elements, but the quadrature points are material points (MaterialPointBase).
To this end, they provide an interface to get the number and locations of the material points, and the
material point type, following a specific quadrature rule and order.
"""

from abc import abstractmethod

import numpy as np
from edelweissfe.points.node import Node

from edelweissmpm.cells.base.cell import CellBase


class CellElementBase(CellBase):
    """The base class for all CellElements.
    If you want to implement a new CellElement, you have to inherit from this class.

    Parameters
    ----------
    cellType
        A string identifying the requested element formulation.
    cellNumber
        A unique integer label used for all kinds of purposes.
    nodes
        The list of nodes assigned to this cell.
    quadratureType
        A string identifying the requested quadrature type.
    quadratureOrder
        The order of the quadrature rule.
    """

    @abstractmethod
    def __init__(
        self, cellElType: str, cellElNumber: int, nodes: list[Node], quadratureType: str, quadratureOrder: int
    ):
        pass

    @property
    @abstractmethod
    def nMaterialPoints(self) -> int:
        """The unique number of this cell"""

    @abstractmethod
    def getRequestedMaterialPointCoordinates(self) -> np.ndarray:
        """Get the list of the requested material point coordinates.

        Returns
        -------
        np.ndarray
            The material point coordinates.
        """

    @abstractmethod
    def getRequestedMaterialPointVolumes(self) -> np.ndarray:
        """Get the list of the requested material point volumes.

        Returns
        -------
        np.ndarray
            The material point volumes.
        """

    @abstractmethod
    def getRequestedMaterialPointType(self) -> type:
        """Get the type of the requested material point.

        Returns
        -------
        type
            The material point type.
        """

    @abstractmethod
    def acceptLastState(self):
        """Accept the last state of the CellElement.

        This method is called after a time step has been accepted.
        """
