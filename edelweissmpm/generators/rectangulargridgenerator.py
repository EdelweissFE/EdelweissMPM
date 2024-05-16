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


def generateModelData(
    model,
    journal,
    name="rectangular_grid",
    x0=0.0,
    y0=0.0,
    h=1.0,
    l=1.0,
    nX=10,
    nY=10,
    firstCellNumber=1,
    firstNodeNumber=1,
    cellProvider=str,
    cellType=str,
    nodesPerCell=4,
):
    """Generate a structured grid of nodes and cells.

    Parameters
    ----------
    model : FEModel
        The model to which the grid should be added.
    journal : Journal
        The journal instance for logging purposes.
    name : str
        The name of the grid.
    x0 : float
        The x-coordinate of the origin.
    y0 : float
        The y-coordinate of the origin.
    h : float
        The height of the grid.
    l : float
        The length of the grid.
    nX : int
        The number of elements along the x-axis.
    nY : int
        The number of elements along the y-axis.
    cellProvider : str
        The name of the cell provider class.
    cellType : str
        The type of the cell.
    firstNodeNumber : int
        The first node number.
    nodesPerCell : int
        The number of nodes per cell.
    """

    CellFactory = getCellClass(cellProvider)

    if nodesPerCell == 4:
        nNodesX = nX + 1
        nNodesY = nY + 1

    elif nodesPerCell == 8:
        nNodesX = 2 * nX + 1
        nNodesY = 2 * nY + 1

    else:
        raise Exception("Invalid number of nodes per grid cell specified")

    grid = np.mgrid[
        x0 : x0 + l : nNodesX * 1j,
        y0 : y0 + h : nNodesY * 1j,
    ]

    nodes = []
    currentNodeNumber = firstNodeNumber

    for x in range(nNodesX):
        for y in range(nNodesY):
            node = Node(currentNodeNumber, grid[:, x, y])
            model.nodes[currentNodeNumber] = node
            nodes.append(node)
            currentNodeNumber += 1

    nG = np.asarray(nodes).reshape(nNodesX, nNodesY)

    currentCellNumber = firstCellNumber

    cells = []
    for x in range(nX):
        for y in range(nY):
            if nodesPerCell == 4:
                cellNodes = [nG[x, y], nG[x + 1, y], nG[x + 1, y + 1], nG[x, y + 1]]
                newCell = CellFactory(cellType, currentCellNumber, cellNodes)

            elif nodesPerCell == 8:
                cellNodes = [
                    nG[2 * x, 2 * y],
                    nG[2 * x + 2, 2 * y],
                    nG[2 * x + 2, 2 * y + 2],
                    nG[2 * x, 2 * y + 2],
                    nG[2 * x + 1, 2 * y],
                    nG[2 * x + 2, 2 * y + 1],
                    nG[2 * x + 1, 2 * y + 2],
                    nG[2 * x, 2 * y + 1],
                ]
                newCell = CellFactory(cellType, currentCellNumber, cellNodes)
            cells.append(newCell)
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

    # if "cellelementProvider" in kwargs:

    #     mpType = kwargs["mpType"]
    #     MPFactory = kwargs["mpClass"]

    #     mpThickness = float(kwargs.get("thickness", 1.0))

    #     currentMPNumber = int(kwargs.get("mpNumberStart", len(model.materialPoints) + 1))
    #     for cell in cells:

    #         nMaterialPoints = cell.nMaterialPoints
    #         mpCoords = cell.getRequestedMaterialPointCoordinates()
    #         mpVolumes = cell.getRequestedMaterialPointVolumes()

    #         for coord, vol in zip(mpCoords, mpVolumes):

    #             while currentMPNumber in model.materialPoints:
    #                 currentMPNumber += 1

    #             mp = MPFactory(mpType, currentMPNumber, coord.reshape((1, -1)), vol)
    #             currentMPNumber += 1

    #             model.materialPoints[currentMPNumber] = mp

    return model
