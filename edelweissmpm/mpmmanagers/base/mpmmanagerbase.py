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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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
Material Point Managers are responsible for the connectivity between material points and cells.
They track the material points and cells and update the connectivity between them.
Furthermore, they provide the active cells, which are the cells that contain material points."""

from edelweissmpm.sets.cellset import CellSet
from edelweissmpm.cells.base.cell import CellBase
from edelweissmpm.materialpoints.base.mp import MaterialPointBase

from collections import defaultdict

import numpy as np

from abc import ABC, abstractmethod


class MPMManagerBase(ABC):
    """
    The MPMManagerBase class is an abstract base class for all material point managers.
    If you want to implement a new material point manager, you have to inherit from this class."""

    @abstractmethod
    def updateConnectivity(
        self,
    ) -> bool:
        """
        Update the connectivity between observed cells and material points.
        Returns the truth value of a change during the last connectivity update.

        This method will:
            - assign all observed material points the cells in which they currently reside.
            - assign all active cells the material points which are residing in the cell.

        Returns
        -------
        bool
            The truth value of change.
        """

        pass

    @abstractmethod
    def getActiveCells(
        self,
    ) -> CellSet:
        """Returns the set of active cells, which have been determined during the last connectivity update.

        Returns
        -------
        CellSet
            The set of active cells.
        """
        pass
