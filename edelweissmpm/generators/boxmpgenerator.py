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

from edelweissfe.journal.journal import Journal

from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.config.mplibrary import getMaterialPointClass
from edelweissmpm.sets.materialpointset import MaterialPointSet

import numpy as np


def generateModelData(model: MPMModel, journal: Journal, **kwargs):
    name = kwargs.get("name", "planeRect")

    x0 = float(kwargs.get("x0", 0.0))
    y0 = float(kwargs.get("y0", 0.0))
    z0 = float(kwargs.get("z0", 0.0))
    l = float(kwargs.get("l", 1.0))
    h = float(kwargs.get("h", 1.0))
    d = float(kwargs.get("d", 1.0))
    nX = int(kwargs.get("nX", 10))
    nY = int(kwargs.get("nY", 10))
    nZ = int(kwargs.get("nZ", 10))
    firstMaterialPointNumber = int(kwargs.get("mpNumberStart", 1))
    mpClass = kwargs["mpProvider"]
    mpType = kwargs["mpType"]
    material = kwargs["material"]

    mpVolume = l * h * d / (nX * nY * nZ)

    MPFactory = getMaterialPointClass(mpClass)

    grid = np.mgrid[
        x0 : x0 + l : nX * 1j,
        y0 : y0 + h : nY * 1j,
        z0 : z0 + d : nZ * 1j,
    ]

    mps = []
    currentMPNumber = firstMaterialPointNumber

    for x in range(nX):
        for y in range(nY):
            for z in range(nZ):
                mpCoordinates = grid[:, x, y, z].reshape(-1, 3)
                mp = MPFactory(mpType, currentMPNumber, mpCoordinates, mpVolume, material)
                model.materialPoints[currentMPNumber] = mp
                mps.append(mp)
                currentMPNumber += 1

    mpGrid = np.asarray(mps).reshape(nX, nY, nZ)

    model.materialPointSets["{:}_left".format(name)] = MaterialPointSet(
        "{:}_left".format(name), [n for n in mpGrid[0, :, :].flatten()]
    )
    model.materialPointSets["{:}_right".format(name)] = MaterialPointSet(
        "{:}_right".format(name), [n for n in mpGrid[-1, :, :].flatten()]
    )
    model.materialPointSets["{:}_top".format(name)] = MaterialPointSet(
        "{:}_top".format(name), [n for n in mpGrid[:, -1, :].flatten()]
    )
    model.materialPointSets["{:}_bottom".format(name)] = MaterialPointSet(
        "{:}_bottom".format(name), [n for n in mpGrid[:, 0, :].flatten()]
    )
    model.materialPointSets["{:}_leftBottom".format(name)] = MaterialPointSet(
        "{:}_leftBottom".format(name), [n for n in mpGrid[0, 0, :]]
    )
    model.materialPointSets["{:}_leftTop".format(name)] = MaterialPointSet(
        "{:}_leftTop".format(name), [n for n in mpGrid[0, -1, :]]
    )

    model.materialPointSets["{:}_rightBottom".format(name)] = MaterialPointSet(
        "{:}_rightBottom".format(name), [n for n in mpGrid[-1, 0, :]]
    )
    model.materialPointSets["{:}_rightTop".format(name)] = MaterialPointSet(
        "{:}_rightTop".format(name), [n for n in mpGrid[-1, -1, :]]
    )

    # model.materialPointSets["{:}_boundary".format(name)] = MaterialPointSet(
    #     "{:}_boundary".format(name),
    #     [n for n in mpGrid[:, -1]]
    #     + [n for n in mpGrid[:, 0]]
    #     + [n for n in mpGrid[0, 1:-1]]
    #     + [n for n in mpGrid[-1, 1:-1]],
    # )

    return model
