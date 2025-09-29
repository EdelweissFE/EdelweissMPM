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
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet

from edelweissmpm.config.celllibrary import getCellClass


def generateModelData(model, journal, **kwargs):
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
    name = kwargs.get("name", "rectangular_grid")

    x0 = float(kwargs.get("x0", 0.0))
    y0 = float(kwargs.get("y0", 0.0))
    hght = float(kwargs.get("h", 1.0))
    lgth = float(kwargs.get("l", 1.0))
    nX = int(kwargs.get("nX", 10))
    nY = int(kwargs.get("nY", 10))
    order = int(kwargs["order"])

    firstNodeNumber = int(kwargs.get("firstNodeNumber", 1))
    firstCellNumber = int(kwargs.get("firstCellNumber", 1))
    cellClass = kwargs["cellProvider"]
    cellType = kwargs["cellType"]

    CellFactory = getCellClass(cellClass)

    nShapeFunctionsPerCellDirection = order + 1
    nKnotsPerCellDirection = (order + 2) + (nShapeFunctionsPerCellDirection - 1)

    nNodesX = nShapeFunctionsPerCellDirection + (nX - 1)
    nNodesY = nShapeFunctionsPerCellDirection + (nY - 1)

    nKnotsX = nKnotsPerCellDirection + (nX - 1)
    nKnotsY = nKnotsPerCellDirection + (nY - 1)

    nodeGrid = np.mgrid[
        x0 : x0 + lgth : nNodesX * 1j,
        y0 : y0 + hght : nNodesY * 1j,
    ]

    knotsX = np.hstack((np.full(order, x0), np.linspace(x0, x0 + lgth, nKnotsX - 2 * order), np.full(order, x0 + lgth)))

    knotsY = np.hstack((np.full(order, y0), np.linspace(y0, y0 + hght, nKnotsY - 2 * order), np.full(order, y0 + hght)))

    nodes = []
    currentNodeNumber = firstNodeNumber

    for x in range(nNodesX):
        for y in range(nNodesY):
            node = Node(currentNodeNumber, nodeGrid[:, x, y])
            model.nodes[currentNodeNumber] = node
            nodes.append(node)
            currentNodeNumber += 1

    nG = np.asarray(nodes).reshape(nNodesX, nNodesY)

    currentCellNumber = firstCellNumber

    for x in range(nX):
        for y in range(nY):
            cellNodes = []
            for j in range(nShapeFunctionsPerCellDirection):
                for i in range(nShapeFunctionsPerCellDirection):
                    cellNodes.append(nG[x + i, y + j])

            cellKnotsX = knotsX[x : x + nKnotsPerCellDirection]
            cellKnotsY = knotsY[y : y + nKnotsPerCellDirection]

            cellKnots = np.array([cellKnotsX, cellKnotsY])
            newCell = CellFactory(cellType, currentCellNumber, cellNodes, cellKnots)

            model.cells[currentCellNumber] = newCell

            currentCellNumber += 1

    model.nodeSets["{:}_left".format(name)] = NodeSet("{:}_left".format(name), [n for n in nG[0, :]])
    model.nodeSets["{:}_right".format(name)] = NodeSet("{:}_right".format(name), [n for n in nG[-1, :]])
    model.nodeSets["{:}_top".format(name)] = NodeSet("{:}_top".format(name), [n for n in nG[:, -1]])
    model.nodeSets["{:}_bottom".format(name)] = NodeSet("{:}_bottom".format(name), [n for n in nG[:, 0]])

    model.nodeSets["{:}_leftBottom".format(name)] = NodeSet("{:}_leftBottom".format(name), [nG[0, 0]])
    model.nodeSets["{:}_leftTop".format(name)] = NodeSet("{:}_leftTop".format(name), [nG[0, -1]])
    model.nodeSets["{:}_rightBottom".format(name)] = NodeSet("{:}_rightBottom".format(name), [nG[-1, 0]])
    model.nodeSets["{:}_rightTop".format(name)] = NodeSet("{:}_rightTop".format(name), [nG[-1, -1]])

    return model
