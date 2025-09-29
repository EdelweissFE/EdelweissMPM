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
#  Thomas Mader thomas.mader@boku.ac.at
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


def generateCooksMembraneKernelFunctionGrid(
    model: MPMModel,
    journal: Journal,
    kernelFunctionFactoryCallback: typing.Callable[[Node], BaseMeshfreeKernelFunction],
    name: str = "cooks_membrane",
    x0: float = 0.0,
    y0: float = 0.0,
    l: float = 48.0,
    h0: float = 44.0,
    h1: float = 16.0,
    nX: int = 10,
    nY: int = 10,
    firstKernelFunctionNumber: int = 1,
):
    """Generate a Cook's membrane-shaped grid of particles (trapezoidal mesh).

    Returns
    -------
    MPMModel
        The updated model.
    """
    x_coords = np.linspace(x0, x0 + l, nX + 1)
    vertices = np.zeros((nX + 1, nY + 1, 2))  # (x, y) coordinates

    for i, x in enumerate(x_coords):
        y_top = y0 + h0 + h1 / l * (x - x0)
        y_bottom = y0 + h0 / l * (x - x0)

        y_coords = np.linspace(y_bottom, y_top, nY + 1)

        for j, y in enumerate(y_coords):
            vertices[i, j] = [x, y]

    currentKernelFunctionNumber = firstKernelFunctionNumber
    nodes = []

    for i in range(nX):
        for j in range(nY):
            quad = np.array(
                [
                    vertices[i, j],
                    vertices[i + 1, j],
                    vertices[i + 1, j + 1],
                    vertices[i, j + 1],
                ]
            )
            pCoordinates = np.mean(quad, axis=0)  # centroid of the quad

            kf = kernelFunctionFactoryCallback(Node(currentKernelFunctionNumber, pCoordinates))
            model.meshfreeKernelFunctions[currentKernelFunctionNumber] = kf
            currentKernelFunctionNumber += 1
            nodes.append(kf.node)

    nG = np.asarray(nodes).reshape(nX, nY)

    # for x in range(nX):
    #    for y in range(nY):
    #        pCoordinates = grid[:, x, y].flatten()

    #        kf = kernelFunctionFactoryCallback(Node(currentKernelFunctionNumber, pCoordinates))

    #        model.meshfreeKernelFunctions[currentKernelFunctionNumber] = kf
    #        currentKernelFunctionNumber += 1
    #        nodes.append(kf.node)

    # nG = np.asarray(nodes).reshape(nX, nY)

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
