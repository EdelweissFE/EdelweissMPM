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

from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.particles.base.baseparticle import BaseParticle
from edelweissmpm.sets.particleset import ParticleSet


def generateRectangularParticleGrid(
    model: MPMModel,
    journal: Journal,
    particleFactoryCallback: typing.Callable[[np.ndarray, float], BaseParticle],
    name: str = "rectangular_grid",
    x0: float = 0.0,
    y0: float = 0.0,
    h: float = 1.0,
    l: float = 1.0,
    nX: int = 10,
    nY: int = 10,
    firstParticleNumber: int = 1,
    thickness: float = 1.0,
):
    """Generate a structured rectangular grid of particles.

    Returns
    -------
    MPMModel
        The updated model.
    """

    pVolume = l * h / (nX * nY) * thickness

    grid = np.mgrid[
        x0 : x0 + l : nX * 1j,
        y0 : y0 + h : nY * 1j,
    ]

    particles = []
    currentParticleNumber = firstParticleNumber

    for x in range(nX):
        for y in range(nY):
            pCoordinates = np.copy(grid[:, x, y]).reshape((-1, 2))
            particle = particleFactoryCallback(currentParticleNumber, pCoordinates, pVolume)

            model.particles[currentParticleNumber] = particle
            particles.append(particle)
            currentParticleNumber += 1

    particleGrid = np.asarray(particles).reshape(nX, nY)

    model.particleSets["{:}_all".format(name)] = ParticleSet("{:}_all".format(name), particleGrid.flatten())

    model.particleSets["{:}_left".format(name)] = ParticleSet("{:}_left".format(name), [n for n in particleGrid[0, :]])
    model.particleSets["{:}_right".format(name)] = ParticleSet(
        "{:}_right".format(name), [n for n in particleGrid[-1, :]]
    )
    model.particleSets["{:}_top".format(name)] = ParticleSet("{:}_top".format(name), [n for n in particleGrid[:, -1]])
    model.particleSets["{:}_bottom".format(name)] = ParticleSet(
        "{:}_bottom".format(name), [n for n in particleGrid[:, 0]]
    )
    model.particleSets["{:}_leftBottom".format(name)] = ParticleSet("{:}_leftBottom".format(name), [particleGrid[0, 0]])
    model.particleSets["{:}_leftTop".format(name)] = ParticleSet("{:}_leftTop".format(name), [particleGrid[0, -1]])
    model.particleSets["{:}_rightBottom".format(name)] = ParticleSet(
        "{:}_rightBottom".format(name), [particleGrid[-1, 0]]
    )
    model.particleSets["{:}_rightTop".format(name)] = ParticleSet("{:}_rightTop".format(name), [particleGrid[-1, -1]])

    model.particleSets["{:}_boundary".format(name)] = ParticleSet(
        "{:}_boundary".format(name),
        [n for n in particleGrid[:, -1]]
        + [n for n in particleGrid[:, 0]]
        + [n for n in particleGrid[0, 1:-1]]
        + [n for n in particleGrid[-1, 1:-1]],
    )

    return model
