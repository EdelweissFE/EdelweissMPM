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
from edelweissfe.journal.journal import Journal

from edelweissmpm.config.mplibrary import getMaterialPointClass
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.sets.materialpointset import MaterialPointSet


def generateModelData(
    model: MPMModel,
    journal: Journal,
    name: str = "rectangular_grid",
    x0: float = 0.0,
    y0: float = 0.0,
    h: float = 1.0,
    l: float = 1.0,
    nX: int = 10,
    nY: int = 10,
    firstMPNumber: int = 1,
    mpProvider: str = None,
    mpType: str = None,
    thickness: float = 1.0,
    material: str = None,
):
    """Generate a structured rectangular grid of material points.

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
    h
        The height of the body.
    l
        The length of the body.
    nX
        The number of material points along x.
    nY
        The number of material points along y.
    firstMPNumber
        The first material point number.
    mpProvider
        The providing class for the MaterialPoint.
    mpType
        The type of MaterialPoint.
    thickness
        The thickness of the material point.
    material
        The material of the material point.

    Returns
    -------
    MPMModel
        The updated model.
    """

    mpVolume = l * h / (nX * nY) * thickness

    MPFactory = getMaterialPointClass(mpProvider)

    grid = np.mgrid[
        x0 : x0 + l : nX * 1j,
        y0 : y0 + h : nY * 1j,
    ]

    mps = []
    currentMPNumber = firstMPNumber

    for x in range(nX):
        for y in range(nY):
            mpCoordinates = grid[:, x, y].reshape(-1, 2)
            mp = MPFactory(mpType, currentMPNumber, mpCoordinates, mpVolume, material)
            model.materialPoints[currentMPNumber] = mp
            mps.append(mp)
            currentMPNumber += 1

    mpGrid = np.asarray(mps).reshape(nX, nY)

    model.materialPointSets["{:}_all".format(name)] = MaterialPointSet("{:}_all".format(name), mpGrid.flatten())

    model.materialPointSets["{:}_left".format(name)] = MaterialPointSet(
        "{:}_left".format(name), [n for n in mpGrid[0, :]]
    )
    model.materialPointSets["{:}_right".format(name)] = MaterialPointSet(
        "{:}_right".format(name), [n for n in mpGrid[-1, :]]
    )
    model.materialPointSets["{:}_top".format(name)] = MaterialPointSet(
        "{:}_top".format(name), [n for n in mpGrid[:, -1]]
    )
    model.materialPointSets["{:}_bottom".format(name)] = MaterialPointSet(
        "{:}_bottom".format(name), [n for n in mpGrid[:, 0]]
    )
    model.materialPointSets["{:}_leftBottom".format(name)] = MaterialPointSet(
        "{:}_leftBottom".format(name), [mpGrid[0, 0]]
    )
    model.materialPointSets["{:}_leftTop".format(name)] = MaterialPointSet("{:}_leftTop".format(name), [mpGrid[0, -1]])
    model.materialPointSets["{:}_rightBottom".format(name)] = MaterialPointSet(
        "{:}_rightBottom".format(name), [mpGrid[-1, 0]]
    )
    model.materialPointSets["{:}_rightTop".format(name)] = MaterialPointSet(
        "{:}_rightTop".format(name), [mpGrid[-1, -1]]
    )

    model.materialPointSets["{:}_boundary".format(name)] = MaterialPointSet(
        "{:}_boundary".format(name),
        [n for n in mpGrid[:, -1]]
        + [n for n in mpGrid[:, 0]]
        + [n for n in mpGrid[0, 1:-1]]
        + [n for n in mpGrid[-1, 1:-1]],
    )

    return model
