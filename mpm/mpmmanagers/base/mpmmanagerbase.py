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

from mpm.sets.cellset import CellSet
from mpm.cells.base.cell import CellBase
from mpm.materialpoints.base.mp import MaterialPointBase

from collections import defaultdict

import numpy as np

from abc import ABC, abstractmethod


class MPMManagerBase(ABC):
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
