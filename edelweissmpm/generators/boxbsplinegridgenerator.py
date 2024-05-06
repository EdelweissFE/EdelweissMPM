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

from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.misc import convertLinesToStringDictionary
from edelweissfe.journal.journal import Journal
from edelweissfe.variables.fieldvariable import FieldVariable

from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.config.celllibrary import getCellClass

import numpy as np


def generateModelData(
    model: MPMModel,
    journal: Journal,
    name: str = "boxgrid",
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    l: float = 1.0,
    h: float = 1.0,
    d: float = 1.0,
    nX: int = 10,
    nY: int = 10,
    nZ: int = 10,
    order: int = 1,
    firstNodeNumber: int = 1,
    cellProvider: type = None,
    cellType: str = None,
):
    """
    Generates a B-Spline grid of cells in a box.

    Parameters
    ----------
    model : MPMModel
        The model to which the grid will be added.
    journal : Journal
        The journal instance for logging purposes.
    name : str
        The name of the grid.
    x0 : float
        The origin at x axis.
    y0 : float
        The origin at y axis.
    z0 : float
        The origin at z axis.
    l : float
        The length of the box.
    h : float
        The height of the box.
    d : float
        The depth of the box.
    nX : int
        The number of cells in x direction.
    nY : int
        The number of cells in y direction.
    nZ : int
        The number of cells in z direction.
    order : int
        The order of the B-Spline basis functions.
    firstNodeNumber : int
        The first node number.
    cellProvider : type
        The providing class for the Cell.
    cellType : str
        The type of Cell.

    Returns
    -------
    MPMModel
        The model with the added grid.
    """

    CellFactory = getCellClass(cellProvider)

    nShapeFunctionsPerCellDirection = order + 1
    nKnotsPerCellDirection = (order + 2) + (nShapeFunctionsPerCellDirection - 1)

    nNodesX = nShapeFunctionsPerCellDirection + (nX - 1)
    nNodesY = nShapeFunctionsPerCellDirection + (nY - 1)
    nNodesZ = nShapeFunctionsPerCellDirection + (nZ - 1)

    nKnotsX = nKnotsPerCellDirection + (nX - 1)
    nKnotsY = nKnotsPerCellDirection + (nY - 1)
    nKnotsZ = nKnotsPerCellDirection + (nZ - 1)

    nodeGrid = np.mgrid[
        x0 : x0 + l : nNodesX * 1j,
        y0 : y0 + h : nNodesY * 1j,
        z0 : z0 + d : nNodesZ * 1j,
    ]

    knotsX = np.hstack((np.full(order, x0), np.linspace(x0, x0 + l, nKnotsX - 2 * order), np.full(order, x0 + l)))
    knotsY = np.hstack((np.full(order, y0), np.linspace(y0, y0 + h, nKnotsY - 2 * order), np.full(order, y0 + h)))
    knotsZ = np.hstack((np.full(order, z0), np.linspace(z0, z0 + d, nKnotsZ - 2 * order), np.full(order, z0 + d)))

    nodes = []
    currentNodeNumber = firstNodeNumber

    for x in range(nNodesX):
        for y in range(nNodesY):
            for z in range(nNodesZ):
                node = Node(currentNodeNumber, nodeGrid[:, x, y, z])
                model.nodes[currentNodeNumber] = node
                nodes.append(node)
                currentNodeNumber += 1

    nG = np.asarray(nodes).reshape(nNodesX, nNodesY, nNodesZ)

    currentCellNumber = 1

    cells = []
    for x in range(nX):
        for y in range(nY):
            for z in range(nZ):
                cellNodes = []
                for k in range(nShapeFunctionsPerCellDirection):
                    for j in range(nShapeFunctionsPerCellDirection):
                        for i in range(nShapeFunctionsPerCellDirection):
                            cellNodes.append(nG[x + i, y + j, z + k])

                cellKnotsX = knotsX[x : x + nKnotsPerCellDirection]
                cellKnotsY = knotsY[y : y + nKnotsPerCellDirection]
                cellKnotsZ = knotsZ[z : z + nKnotsPerCellDirection]

                cellKnots = np.array([cellKnotsX, cellKnotsY, cellKnotsZ])
                newCell = CellFactory(cellType, currentCellNumber, cellNodes, cellKnots)

                model.cells[currentCellNumber] = newCell

                currentCellNumber += 1

    model.nodeSets["{:}_left".format(name)] = NodeSet("{:}_left".format(name), [n for n in nG[0, :, :].flatten()])
    model.nodeSets["{:}_right".format(name)] = NodeSet("{:}_right".format(name), [n for n in nG[-1, :, :].flatten()])
    model.nodeSets["{:}_top".format(name)] = NodeSet("{:}_top".format(name), [n for n in nG[:, -1, :].flatten()])
    model.nodeSets["{:}_bottom".format(name)] = NodeSet("{:}_bottom".format(name), [n for n in nG[:, 0, :].flatten()])
    model.nodeSets["{:}_front".format(name)] = NodeSet("{:}_front".format(name), nG[:, :, -1].flatten())
    model.nodeSets["{:}_back".format(name)] = NodeSet("{:}_back".format(name), nG[:, :, 0].flatten())

    model.nodeSets["{:}_leftBottom".format(name)] = NodeSet("{:}_leftBottom".format(name), nG[0, 0, :].flatten())
    model.nodeSets["{:}_leftTop".format(name)] = NodeSet("{:}_leftTop".format(name), nG[0, -1, :].flatten())
    model.nodeSets["{:}_rightBottom".format(name)] = NodeSet("{:}_rightBottom".format(name), nG[-1, 0, :].flatten())
    model.nodeSets["{:}_rightTop".format(name)] = NodeSet("{:}_rightTop".format(name), nG[-1, -1, :].flatten())

    return model
