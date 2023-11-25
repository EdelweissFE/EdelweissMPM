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
    "provider": "The providing class for the MaterialPoint",
    "type": "type of MaterialPoint",
}

# from fe.points.node import MaterialPoint
# from fe.sets.nodeset import MaterialPointSet
# from fe.utils.misc import convertLinesToStringDictionary
from fe.journal.journal import Journal

from mpm.models.mpmmodel import MPMModel
from mpm.config.mplibrary import getMaterialPointClass
from mpm.sets.materialpointset import MaterialPointSet

import numpy as np


def generateModelData(model: MPMModel, journal: Journal, **kwargs):
    name = kwargs.get("name", "planeRect")

    x0 = float(kwargs.get("x0", 0.0))
    y0 = float(kwargs.get("y0", 0.0))
    h = float(kwargs.get("h", 1.0))
    l = float(kwargs.get("l", 1.0))
    nX = int(kwargs.get("nX", 10))
    nY = int(kwargs.get("nY", 10))
    firstMaterialPointLabel = int(kwargs.get("mpLabelStart", 1))
    mpClass = kwargs["mpProvider"]
    mpType = kwargs["mpType"]
    mpThickness = float(kwargs.get("thickness", 1.0))

    mpVolume = l * h / (nX * nY) * mpThickness

    MPFactory = getMaterialPointClass(mpClass)

    grid = np.mgrid[
        x0 : x0 + l : nX * 1j,
        y0 : y0 + h : nY * 1j,
    ]

    mps = []
    currentMPLabel = firstMaterialPointLabel

    for x in range(nX):
        for y in range(nY):
            mpCoordinates = grid[:, x, y].reshape(-1, 2)
            mp = MPFactory(mpType, currentMPLabel, mpCoordinates, mpVolume)
            model.materialPoints[currentMPLabel] = mp
            mps.append(mp)
            currentMPLabel += 1

    mpGrid = np.asarray(mps).reshape(nX, nY)

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

    return model
