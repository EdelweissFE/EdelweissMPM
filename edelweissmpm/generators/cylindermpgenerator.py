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
    name = kwargs.get("name", "cylinder")

    x0 = float(kwargs.get("x0", 0.0))
    y0 = float(kwargs.get("y0", 0.0))
    z0 = float(kwargs.get("z0", 0.0))
    distance = float(kwargs.get("distance", 0.0))
    maxR = float(kwargs.get("R", 0.0))
    H = float(kwargs.get("H", 0.0))
    nZ = int(kwargs.get("nZ", H / distance))
    angle = float(kwargs.get("angle", 0.0))

    firstMaterialPointNumber = int(kwargs.get("mpNumberStart", 1))
    mpClass = kwargs["mpProvider"]
    mpType = kwargs["mpType"]

    MPFactory = getMaterialPointClass(mpClass)

    nCircles = int(np.ceil(maxR / distance))

    Radii = np.linspace(0.0, maxR, nCircles)
    T = [int(np.ceil(angle * r / distance)) for r in Radii]
    heights = np.linspace(0, H, nZ)

    exactV = np.pi * maxR**2 * angle / (np.pi * 2)
    numV = 0.0
    mps = []
    currentMPNumber = firstMaterialPointNumber

    sleeveMPs = []
    frontMPs = []
    backMPs = []
    frontRightMP = []
    for j, h in enumerate(heights):
        for i, (radius, nPoints) in enumerate(zip(Radii, T)):
            if not nPoints:
                continue

            mpHeight = H / nZ
            mpVolume = mpHeight * (angle * radius) * distance / nPoints
            numV += mpVolume * nPoints

            t = np.linspace(0, angle, nPoints, endpoint=True)

            for k, t_ in enumerate(t):
                mpCoordinates = np.array([radius * np.cos(t_) + x0, radius * np.sin(t_) + y0, h + z0]).reshape((1, -1))

                mp = MPFactory(mpType, currentMPNumber, mpCoordinates, mpVolume)
                model.materialPoints[currentMPNumber] = mp
                mps.append(mp)
                currentMPNumber += 1

                if i == nCircles - 1:
                    sleeveMPs.append(mp)

                if j == 0:
                    backMPs.append(mp)

                if j == len(heights) - 1:
                    frontMPs.append(mp)

                    if i == nCircles - 1 and k == 0:
                        frontRightMP.append(mp)

    model.materialPointSets["{:}_sleeve".format(name)] = MaterialPointSet("{:}_sleeve".format(name), sleeveMPs)
    model.materialPointSets["{:}_front".format(name)] = MaterialPointSet("{:}_front".format(name), frontMPs)
    model.materialPointSets["{:}_back".format(name)] = MaterialPointSet("{:}_back".format(name), backMPs)
    model.materialPointSets["{:}_frontRight".format(name)] = MaterialPointSet(
        "{:}_frontRight".format(name), frontRightMP
    )

    return model
