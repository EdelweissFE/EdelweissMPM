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

from edelweissfe.utils.meshtools import extractNodesFromElementSet


class CellSet(list):
    """A basic element set.
    It has a label, and a list containing unique cells.

    Parameters
    ----------
    name
        The unique label for this element set.
    cells
        The list of cells.
    """

    # def __new__(cls, name, cells):
    #     instance = super().__new__(cls, cells)
    #     instance.name = name
    #     instance._nodes = None
    #     return instance

    # allowedObjectTypes = [CellBase]

    def __init__(
        self,
        name: str,
        elements: list,
    ):
        self.name = name
        super().__init__(elements)
        self._nodes = None

    def __hash__(self):
        return hash(self.name)

    def extractNodeSet(
        self,
    ):
        if not self._nodes:
            self._nodes = extractNodesFromElementSet(self)
        return self._nodes
