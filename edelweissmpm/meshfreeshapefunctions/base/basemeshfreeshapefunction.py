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
Meshfree shapefunctions are one of the core ingredients in meshfree methods.

"""

from abc import ABC, abstractmethod

import numpy as np
from edelweissfe.points.node import Node


class BaseMeshfreeShapeFunction(ABC):
    """Base class for meshfree shape functions.
    Shape functions are used to interpolate field solutions from the nodes to arbitrary points in the domain.
    Furthermore, they are the link between the material points and the nodes in meshfree methods.

    Each shape function should be derived from this base class in order to follow the general interface.
    """

    @property
    @abstractmethod
    def node(self) -> Node:
        """Get the node of the shape function.

        Returns
        -------
        Node
            The node of the shape function.
        """

    @abstractmethod
    def getCurrentBoundingBox(self) -> np.ndarray:
        """Get the bounding box of the shape function.

        Returns
        -------
        np.ndarray
            The bounding box of the shape function.
        """

    @abstractmethod
    def isCoordinateInCurrentSupport(self, coords: np.ndarray) -> bool:
        """Check if the given coordinates are in the support of the shape function.

        Parameters
        ----------
        coords
            The coordinates to check.

        Returns
        -------
        bool
            True if the coordinates are in the support of the shape function, False otherwise.
        """
