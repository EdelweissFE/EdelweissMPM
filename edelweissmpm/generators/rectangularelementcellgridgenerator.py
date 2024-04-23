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

from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.misc import convertLinesToStringDictionary
from edelweissfe.models.femodel import FEModel
from edelweissfe.journal.journal import Journal
from edelweissfe.variables.fieldvariable import FieldVariable

from edelweissmpm.config.celllibrary import getCellClass
from edelweissmpm.config.cellelementlibrary import getCellElementClass

import numpy as np


def generateModelData(model, journal, **kwargs):
    """Generate a structured grid of nodes and cells.

    Parameters
    ----------
    model : FEModel
        The model to which the grid should be added.
    journal : Journal
        The journal instance for logging purposes.

    Other Parameters
    ----------------
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
    firstNodeNumber : int
        The first node number.
    nodesPerCell : int
        The number of nodes per cell.
    mpType : str
        The type of the material point.
    mpClass : str
        The name of the material point class.
    thickness : float
        The thickness of the material point.
    mpNumberStart : int
        The first material point number.
    cellelementProvider : str
        The name of the cell element provider class.
    cellelementType : str
        The type of the cell element.
    """

    name = kwargs.get("name", "rectangular_grid")

    x0 = float(kwargs.get("x0", 0.0))
    y0 = float(kwargs.get("y0", 0.0))
    h = float(kwargs.get("h", 1.0))
    l = float(kwargs.get("l", 1.0))
    nX = int(kwargs.get("nX", 10))
    nY = int(kwargs.get("nY", 10))
    firstNodeNumber = int(kwargs.get("cellNumberStart", 1))

    cellClass = kwargs["cellelementProvider"]
    cellType = kwargs["cellelementType"]
    CellFactory = getCellElementClass(cellClass)

    nodesPerCell = int(kwargs.get("nodesPerCell", 4))

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

    currentCellNumber = 1

    cellElements = []
    for x in range(nX):
        for y in range(nY):
            if nodesPerCell == 4:
                cellNodes = [nG[x, y], nG[x + 1, y], nG[x + 1, y + 1], nG[x, y + 1]]
                newCellElement = CellFactory(cellType, currentCellNumber, cellNodes)

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
                newCellElement = CellFactory(cellType, currentCellNumber, cellNodes)
            cellElements.append(newCellElement)
            model.elements[currentCellNumber] = newCellElement

            currentCellNumber += 1

    model.nodeSets["{:}_left".format(name)] = NodeSet("{:}_left".format(name), [n for n in nG[0, :]])
    model.nodeSets["{:}_right".format(name)] = NodeSet("{:}_right".format(name), [n for n in nG[-1, :]])
    model.nodeSets["{:}_top".format(name)] = NodeSet("{:}_top".format(name), [n for n in nG[:, -1]])
    model.nodeSets["{:}_bottom".format(name)] = NodeSet("{:}_bottom".format(name), [n for n in nG[:, 0]])

    model.nodeSets["{:}_leftBottom".format(name)] = NodeSet("{:}_leftBottom".format(name), [nG[0, 0]])
    model.nodeSets["{:}_leftTop".format(name)] = NodeSet("{:}_leftTop".format(name), [nG[0, -1]])
    model.nodeSets["{:}_rightBottom".format(name)] = NodeSet("{:}_rightBottom".format(name), [nG[-1, 0]])
    model.nodeSets["{:}_rightTop".format(name)] = NodeSet("{:}_rightTop".format(name), [nG[-1, -1]])

    mpType = kwargs["mpType"]
    MPFactory = kwargs["mpClass"]

    mpThickness = float(kwargs.get("thickness", 1.0))

    currentMPNumber = int(kwargs.get("mpNumberStart", len(model.materialPoints) + 1))
    for cell in cellElements:

        nMaterialPoints = cell.nMaterialPoints
        mpCoords = cell.getRequestedMaterialPointCoordinates()
        mpVolumes = cell.getRequestedMaterialPointVolumes()

        cellMPs = []
        for coord, vol in zip(mpCoords, mpVolumes):

            while currentMPNumber in model.materialPoints:
                currentMPNumber += 1

            mp = MPFactory(
                mpType,
                currentMPNumber,
                coord.reshape((1, 2)),
                vol,
            )
            mp.assignMaterial("GMDAMAGEDSHEARNEOHOOKE", np.array([30000.0, 0.3, 1.0, 2, 4, 1.4999]))
            mp.initializeYourself()

            model.materialPoints[currentMPNumber] = mp
            cellMPs.append(mp)

            currentMPNumber += 1

        cell.assignMaterialPoints(cellMPs)

    return model
