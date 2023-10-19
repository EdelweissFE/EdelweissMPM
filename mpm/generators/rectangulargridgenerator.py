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

A mesh generator, for rectangular geometries and structured quad meshes:


.. code-block:: console

        <-----l----->
         nX elements
         __ __ __ __
        |__|__|__|__|  A
        |__|__|__|__|  |
        |__|__|__|__|  | h
        |__|__|__|__|  | nY elements
      | |__|__|__|__|  |
      | |__|__|__|__|  V
    x0|_____
      y0
  
nSets, elSets, surface : 'name'_top, _bottom, _left, _right, ...
are automatically generated

Datalines:
"""

documentation = {
    "x0": "(optional) origin at x axis",
    "y0": "(optional) origin at y axis",
    "h": "(optional) height of the body",
    "l": "(optional) length of the body",
    "nX": "(optional) number of elements along x",
    "nY": "(optional) number of elements along y",
    "elType": "type of element",
}

from fe.points.node import Node
from fe.sets.nodeset import NodeSet
from fe.utils.misc import convertLinesToStringDictionary
from fe.models.femodel import FEModel
from fe.journal.journal import Journal
from fe.variables.fieldvariable import FieldVariable

from mpm.config.celllibrary import getCellClass

import numpy as np


def generateModelData(model, journal, **kwargs):
    # options = generatorDefinition["data"]
    # options = convertLinesToStringDictionary(options)

    name = kwargs.get("name", "planeRect")

    x0 = float(kwargs.get("x0", 0.0))
    y0 = float(kwargs.get("y0", 0.0))
    h = float(kwargs.get("h", 1.0))
    l = float(kwargs.get("l", 1.0))
    nX = int(kwargs.get("nX", 10))
    nY = int(kwargs.get("nY", 10))
    firstNodeLabel = int(kwargs.get("cellLabelStart", 1))
    cellClass = kwargs["cellProvider"]
    cellType = kwargs["cellType"]

    CellFactory = getCellClass(cellClass)

    testCell = CellFactory(
        cellType,
        0,
    )
    if testCell.nNodes == 4:
        nNodesX = nX + 1
        nNodesY = nY + 1

    if testCell.nNodes == 8:
        nNodesX = 2 * nX + 1
        nNodesY = 2 * nY + 1

    grid = np.mgrid[
        x0 : x0 + l : nNodesX * 1j,
        y0 : y0 + h : nNodesY * 1j,
    ]

    nodes = []
    currentNodeLabel = firstNodeLabel

    for x in range(nNodesX):
        for y in range(nNodesY):
            node = Node(currentNodeLabel, grid[:, x, y])
            model.nodes[currentNodeLabel] = node
            nodes.append(node)
            currentNodeLabel += 1

    nG = np.asarray(nodes).reshape(nNodesX, nNodesY)

    currentCellLabel = 1

    cells = []
    for x in range(nX):
        for y in range(nY):
            if testCell.nNodes == 4:
                newCell = CellFactory(cellType, currentCellLabel)
                newCell.setNodes([nG[x, y], nG[x + 1, y], nG[x + 1, y + 1], nG[x, y + 1]])

            elif testCell.nNodes == 8:
                newCell = CellFactory(
                    cellType,
                    currentCellLabel,
                )
                newCell.setNodes(
                    [
                        nG[2 * x, 2 * y],
                        nG[2 * x + 2, 2 * y],
                        nG[2 * x + 2, 2 * y + 2],
                        nG[2 * x, 2 * y + 2],
                        nG[2 * x + 1, 2 * y],
                        nG[2 * x + 2, 2 * y + 1],
                        nG[2 * x + 1, 2 * y + 2],
                        nG[2 * x, 2 * y + 1],
                    ]
                )
            cells.append(newCell)
            model.cells[currentCellLabel] = newCell

            currentCellLabel += 1

    # model._populateNodeFieldVariablesFromElements()

    # nodesets:
    # model.nodeSets["{:}_all".format(name)] = NodeSet("{:}_all".format(name), list())
    # for n in np.ravel(nG):
    #     if len(n.fields):
    #         model.nodeSets["{:}_all".format(name)].add([n])

    # model.nodeSets["{:}_left".format(name)] = NodeSet("{:}_left".format(name), [n for n in nG[0, :]])
    # model.nodeSets["{:}_right".format(name)] = NodeSet("{:}_right".format(name), [n for n in nG[-1, :]])
    # model.nodeSets["{:}_top".format(name)] = NodeSet("{:}_top".format(name), [n for n in nG[:, -1]])
    # model.nodeSets["{:}_bottom".format(name)] = NodeSet("{:}_bottom".format(name), [n for n in nG[:, 0]])

    # model.nodeSets["{:}_leftBottom".format(name)] = NodeSet("{:}_leftBottom".format(name), [nG[0, 0]])
    # model.nodeSets["{:}_leftTop".format(name)] = NodeSet("{:}_leftTop".format(name), [nG[0, -1]])
    # model.nodeSets["{:}_rightBottom".format(name)] = NodeSet("{:}_rightBottom".format(name), [nG[-1, 0]])
    # model.nodeSets["{:}_rightTop".format(name)] = NodeSet("{:}_rightTop".format(name), [nG[-1, -1]])

    # cellGrid = np.asarray(elements).reshape(nX, nY)

    return model
