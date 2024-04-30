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
from edelweissfe.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissfe.outputmanagers.ensight import (
    createUnstructuredPartFromElementSet,
    EnsightChunkWiseCase,
    EnsightGeometry,
    EnsightUnstructuredPart,
)

import os
import datetime
import numpy as np
from collections import defaultdict, OrderedDict
from distutils.util import strtobool
from edelweissfe.points.node import Node
from edelweissfe.utils.meshtools import disassembleElsetToEnsightShapes
import edelweissfe.config.phenomena
from edelweissfe.utils.math import evalModelAccessibleExpression
from io import TextIOBase
from edelweissmpm.sets.materialpointset import MaterialPointSet
from edelweissmpm.sets.cellset import CellSet
from edelweissmpm.sets.cellelementset import CellElementSet


def createUnstructuredPartFromCellSet(cellPartName, cells: list, partID: int):
    """Determines the cell and node list for an Ensightpart from an
    cell set. The reduced, unique node set is generated, as well as
    the cell to node index mapping for the ensight part.

    Parameters
    ----------
    cellSet
        The list of cells defining this part.
    partID
        The id of this part.
    """

    nodeCounter = 0
    partNodes = dict()
    cellDict = dict()
    for cell in cells:
        cellShape = cell.ensightType
        if cellShape not in cellDict:
            cellDict[cellShape] = dict()
        cellNodeIndices = []
        for node in cell.nodes:
            # if the node is already in the dict, get its index,
            # else insert it, and get the current idx = counter. increase the counter
            idx = partNodes.setdefault(node, nodeCounter)
            cellNodeIndices.append(idx)
            if idx == nodeCounter:
                # the node was just inserted, so increase the counter of inserted nodes
                nodeCounter += 1
        cellDict[cellShape][cell.cellNumber] = cellNodeIndices

    return EnsightUnstructuredPart(cellPartName, partID, partNodes.keys(), cellDict)


def createUnstructuredPartFromMaterialPointSet(mpPartName, mps: list, partID: int):
    """Determines the mp and node list for an Ensightpart from an
    mp set. The reduced, unique node set is generated, as well as
    the mp to node index mapping for the ensight part.

    Parameters
    ----------
    mpSet
        The list of mps defining this part.
    partID
        The id of this part.
    """

    nodeCounter = 0
    mpDict = dict()

    partNodes = list()

    for mp in mps:
        mpShape = mp.ensightType
        if mpShape not in mpDict:
            mpDict[mpShape] = dict()
        mpNodeIndices = []

        for vertexCoord in mp.getVertexCoordinates():
            partNodes.append(Node(nodeCounter, vertexCoord))
            mpNodeIndices.append(nodeCounter)
            nodeCounter += 1

        mpDict[mpShape][mp.number] = mpNodeIndices

    return EnsightUnstructuredPart(mpPartName, partID, partNodes, mpDict)


class OutputManager(EnsightOutputManager):
    identification = "Ensight Export"

    def __init__(self, name, model, fieldOutputController, journal, plotter, **kwargs):
        self._exportCellSetParts = kwargs.get("exportCellSetParts", True)
        self._exportCellElementSetParts = kwargs.get("exportCellElementSetParts", True)
        self._exportMPSetParts = kwargs.get("exportMPSetParts", True)

        self.mpSetToEnsightPart = dict()
        self.cellSetToEnsightPart = dict()
        return super().__init__(name, model, fieldOutputController, journal, plotter)

    def _createGeometryParts(self, firstPartID: int):
        feModelParts = super()._createGeometryParts(firstPartID)

        partCounter = len(feModelParts) + 1

        if self._exportCellSetParts:
            for setName, cellSet in self.model.cellSets.items():
                self.cellSetToEnsightPart[setName] = createUnstructuredPartFromCellSet(
                    "CELLSET_{:}".format(setName), cellSet, partCounter
                )
                feModelParts.append(self.cellSetToEnsightPart[setName])
                partCounter += 1

        if self._exportCellElementSetParts:
            for setName, cellSet in self.model.cellElementSets.items():
                self.cellSetToEnsightPart[setName] = createUnstructuredPartFromCellSet(
                    "CELLELEMENTSET_{:}".format(setName), cellSet, partCounter
                )
                feModelParts.append(self.cellSetToEnsightPart[setName])
                partCounter += 1

        if self._exportMPSetParts:
            for setName, mpSet in self.model.materialPointSets.items():
                self.mpSetToEnsightPart[setName] = createUnstructuredPartFromMaterialPointSet(
                    "MPSET_{:}".format(setName), mpSet, partCounter
                )
                feModelParts.append(self.mpSetToEnsightPart[setName])
                partCounter += 1

        return feModelParts

    def _getTargetPartForFieldOutput(self, fieldOutput, **kwargs):
        if "mpSet" in kwargs:
            return self.mpSetToEnsightPart[kwargs.pop("mpSet")]
        if "cellSet" in kwargs:
            return self.cellSetToEnsightPart[kwargs.pop("cellSet")]
        if "cellElementSet" in kwargs:
            return self.cellSetToEnsightPart[kwargs.pop("cellSet")]

        theSetName = fieldOutput.associatedSet.name

        if isinstance(fieldOutput.associatedSet, MaterialPointSet):
            return self.mpSetToEnsightPart[theSetName]

        if isinstance(fieldOutput.associatedSet, CellSet):
            return self.cellSetToEnsightPart[theSetName]

        if isinstance(fieldOutput.associatedSet, CellElementSet):
            return self.cellSetToEnsightPart[theSetName]

        return super()._getTargetPartForFieldOutput(fieldOutput, **kwargs)
