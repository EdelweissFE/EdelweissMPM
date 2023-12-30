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
from mpm.models.mpmmodel import MPMModel
from mpm.sets.materialpointset import MaterialPointSet
from fe.journal.journal import Journal


def generateModelData(mpmModel: MPMModel, journal: Journal, coordinates: np.ndarray, storeIn: str):
    """Find the mateiral point closest to a given coordinate and store it in a material point set in the model.

    Parameters
    ----------
    mpmModel
        The model instance.
    journal
        The Journal instance for logging purposes.
    coordinates
        The coordinates for which the closest material point should be found.
    storeIn
        The name of the material point set, in which the found material point is stored.

    Returns
    -------
    MPMModel
        The updated model.
    """

    allCoords = np.asarray([mp.getCenterCoordinates() for mp in mpmModel.materialPoints.values()])

    differenceNormLeft = np.linalg.norm(allCoords - coordinates, axis=1)
    indexClosest = differenceNormLeft.argmin()
    closestMP = list(mpmModel.materialPoints.values())[indexClosest]

    mpmModel.materialPointSets[storeIn] = MaterialPointSet(storeIn, [closestMP])

    journal.message(
        "Material point '{:}' ({:}) at {:} is closest to {:}".format(
            storeIn, closestMP.label, closestMP.getCenterCoordinates(), coordinates
        ),
        "findClosestMP",
    )

    return mpmModel
