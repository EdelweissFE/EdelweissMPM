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
import numpy as np
from edelweissfe.fields.nodefield import NodeField, NodeFieldSubset
from edelweissfe.points.node import Node

from edelweissmpm.sets.cellelementset import CellElementSet
from edelweissmpm.sets.cellset import CellSet


class MPMNodeFieldSubset:
    pass


class MPMNodeField(NodeField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _getNodeFieldSubsetClass(self):
        return MPMNodeFieldSubset


class MPMNodeFieldSubset(NodeFieldSubset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _getSubsetNodes(self, subset) -> list[Node]:
        if type(subset) == CellSet or type(subset) == CellElementSet:
            nodeCandidates = subset.extractNodeSet()
            return [n for n in nodeCandidates if n in self.parentNodeField._indicesOfNodesInArray]

        else:
            return super()._getSubsetNodes(subset)
