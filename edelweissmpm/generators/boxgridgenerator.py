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
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissfe.journal.journal import Journal

from edelweissmpm.config.celllibrary import getCellClass

import numpy as np


def generateModelData(
    model: MPMModel,
    journal: Journal,
    name: str = "box_grid",
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    h: float = 1.0,
    l: float = 1.0,
    d: float = 1.0,
    nX: int = 10,
    nY: int = 10,
    nZ: int = 10,
    firstNodeNumber: int = 1,
    cellClass: str = None,
    cellType: str = None,
    nodesPerCell: int = 8,
):
    """Generate a structured box grid of material points.

    Parameters
    ----------
    model
        The model instance.
    journal
        The Journal instance for logging purposes.
    name
        The name of the mesh.
    x0
        The origin at x axis.
    y0
        The origin at y axis.
    z0
        The origin at z axis.
    h
        The height of the body.
    l
        The length of the body.
    d
        The depth of the body.
    nX
        The number of material points along x.
    nY
        The number of material points along y.
    nZ
        The number of material points along z.
    firstNodeNumber
        The first node number.
    cellClass
        The providing class for the Cell.
    cellType
        The type of Cell.

    Returns
    -------
    MPMModel
        The updated model.
    """

    CellFactory = getCellClass(cellClass)

    if nodesPerCell == 8:
        nNodesX = nX + 1
        nNodesY = nY + 1
        nNodesZ = nZ + 1

    else:
        raise Exception("Invalid number of nodes per grid cell specified")

    grid = np.mgrid[
        x0 : x0 + l : nNodesX * 1j,
        y0 : y0 + h : nNodesY * 1j,
        z0 : z0 + d : nNodesZ * 1j,
    ]

    nodes = []
    currentNodeNumber = firstNodeNumber

    for x in range(nNodesX):
        for y in range(nNodesY):
            for z in range(nNodesZ):
                node = Node(currentNodeNumber, grid[:, x, y, z])
                model.nodes[currentNodeNumber] = node
                nodes.append(node)
                currentNodeNumber += 1

    nG = np.asarray(nodes).reshape(nNodesX, nNodesY, nNodesZ)

    currentCellNumber = 1

    cells = []
    for x in range(nX):
        for y in range(nY):
            for z in range(nZ):
                if nodesPerCell == 8:
                    cellNodes = [
                        nG[x, y, z],
                        nG[x + 1, y, z],
                        nG[x + 1, y + 1, z],
                        nG[x, y + 1, z],
                        nG[x, y, z + 1],
                        nG[x + 1, y, z + 1],
                        nG[x + 1, y + 1, z + 1],
                        nG[x, y + 1, z + 1],
                    ]

                    newCell = CellFactory(cellType, currentCellNumber, cellNodes)

                cells.append(newCell)
                model.cells[currentCellNumber] = newCell

                currentCellNumber += 1

    model.nodeSets["{:}_left".format(name)] = NodeSet("{:}_left".format(name), [n for n in nG[0, :, :].flatten()])
    model.nodeSets["{:}_right".format(name)] = NodeSet("{:}_right".format(name), [n for n in nG[-1, :, :].flatten()])
    model.nodeSets["{:}_top".format(name)] = NodeSet("{:}_top".format(name), [n for n in nG[:, -1, :].flatten()])
    model.nodeSets["{:}_bottom".format(name)] = NodeSet("{:}_bottom".format(name), [n for n in nG[:, 0, :].flatten()])

    model.nodeSets["{:}_leftBottom".format(name)] = NodeSet(
        "{:}_leftBottom".format(name), [n for n in nG[0, 0, :].flatten()]
    )
    model.nodeSets["{:}_leftTop".format(name)] = NodeSet(
        "{:}_leftTop".format(name), [n for n in nG[0, -1, :].flatten()]
    )
    model.nodeSets["{:}_rightBottom".format(name)] = NodeSet(
        "{:}_rightBottom".format(name), [n for n in nG[-1, 0, :].flatten()]
    )
    model.nodeSets["{:}_rightTop".format(name)] = NodeSet(
        "{:}_rightTop".format(name), [n for n in nG[-1, -1, :].flatten()]
    )

    return model
