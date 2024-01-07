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
from mpm.models.mpmmodel import MPMModel
from fe.journal.journal import Journal

from mpm.config.celllibrary import getCellClass

import numpy as np


def generateModelData(model: MPMModel, journal: Journal, **kwargs):
    name = kwargs.get("name", "boxgrid")

    x0 = float(kwargs.get("x0", 0.0))
    y0 = float(kwargs.get("y0", 0.0))
    z0 = float(kwargs.get("z0", 0.0))
    h = float(kwargs.get("h", 1.0))
    l = float(kwargs.get("l", 1.0))
    d = float(kwargs.get("d", 1.0))
    nX = int(kwargs.get("nX", 10))
    nY = int(kwargs.get("nY", 10))
    nZ = int(kwargs.get("nZ", 10))
    firstNodeNumber = int(kwargs.get("cellNumberStart", 1))
    cellClass = kwargs["cellProvider"]
    cellType = kwargs["cellType"]
    nodesPerCell = int(kwargs.get("nodesPerCell", 8))

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
