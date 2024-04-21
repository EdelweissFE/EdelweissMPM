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
the abstract base class :class:`~CellBase`.
"""

from abc import ABC, abstractmethod
import numpy as np
from edelweissmpm.points.gridnode import GridNode
from edelweissfe.cells.base.cell import CellBase

from edelweissmpm.materialpoints.base.mp import MaterialPointBase


class CellElementBase(CellBase):
    @abstractmethod
    def __init__(self, cellElType: str, cellElNumber: int, gridnodes: list[GridNode]):
        """MPM cells in EdelweissFE should be derived from this
        base class in order to follow the general interface.

        EdelweissFE expects the layout of the internal and external load vectors, P, PExt, (and the stiffness)
        to be of the form

        .. code-block:: console

            [ gridnode 1 - dofs field 1,
              gridnode 1 - dofs field 2,
              gridnode 1 - ... ,
              gridnode 1 - dofs field n,
              gridnode 2 - dofs field 1,
              ... ,
              gridnode N - dofs field n].

        Parameters
        ----------
        cellType
            A string identifying the requested element formulation.
        cellNumber
            A unique integer label used for all kinds of purposes.
        gridnodes
            The list of GridNodes assigned to this cell.
        """

        pass

    @property
    @abstractmethod
    def nMaterialPoints(self) -> int:
        """The unique number of this cell"""

        pass

    @abstractmethod
    def getRequestedMaterialPointCoordinates(self) -> np.ndarray:
        """Get the list of the requested material point coordinates.

        Returns
        -------
        np.ndarray
            The material point coordinates.
        """
        pass

    @abstractmethod
    def getRequestedMaterialPointVolumes(self) -> np.ndarray:
        """Get the list of the requested material point volumes.

        Returns
        -------
        np.ndarray
            The material point volumes.
        """

        pass
