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

import typing

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet

from edelweissmpm.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)
from edelweissmpm.models.mpmmodel import MPMModel


def generateRectangularKernelFunctionGrid(
    model: MPMModel,
    journal: Journal,
    kernelFunctionFactoryCallback: typing.Callable[[Node], BaseMeshfreeKernelFunction],
    name: str = "rectangular_grid",
    x0: float = 0.0,
    y0: float = 0.0,
    h: float = 1.0,
    l: float = 1.0,
    nX: int = 10,
    nY: int = 10,
    firstKernelFunctionNumber: int = 1,
):
    """Generate a structured rectangular grid of particles.

    Returns
    -------
    MPMModel
        The updated model.
    """

    grid = np.mgrid[
        x0 : x0 + l : nX * 1j,
        y0 : y0 + h : nY * 1j,
    ]

    currentKernelFunctionNumber = firstKernelFunctionNumber

    nodes = []
    for x in range(nX):
        for y in range(nY):
            pCoordinates = grid[:, x, y].flatten()

            kf = kernelFunctionFactoryCallback(Node(currentKernelFunctionNumber, pCoordinates))

            model.meshfreeKernelFunctions[currentKernelFunctionNumber] = kf
            currentKernelFunctionNumber += 1
            nodes.append(kf.node)

    nG = np.asarray(nodes).reshape(nX, nY)

    # check if any of the node labels already exist in the model
    for n in nodes:
        if n.label in model.nodes:
            raise ValueError("Node with label {:} already exists in model.".format(n.label))

    model.nodes.update({n.label: n for n in nodes})
    model.nodeSets["{:}_left".format(name)] = NodeSet("{:}_left".format(name), [n for n in nG[0, :]])
    model.nodeSets["{:}_right".format(name)] = NodeSet("{:}_right".format(name), [n for n in nG[-1, :]])
    model.nodeSets["{:}_top".format(name)] = NodeSet("{:}_top".format(name), [n for n in nG[:, -1]])
    model.nodeSets["{:}_bottom".format(name)] = NodeSet("{:}_bottom".format(name), [n for n in nG[:, 0]])

    model.nodeSets["{:}_leftBottom".format(name)] = NodeSet("{:}_leftBottom".format(name), [nG[0, 0]])
    model.nodeSets["{:}_leftTop".format(name)] = NodeSet("{:}_leftTop".format(name), [nG[0, -1]])
    model.nodeSets["{:}_rightBottom".format(name)] = NodeSet("{:}_rightBottom".format(name), [nG[-1, 0]])
    model.nodeSets["{:}_rightTop".format(name)] = NodeSet("{:}_rightTop".format(name), [nG[-1, -1]])

    return model
