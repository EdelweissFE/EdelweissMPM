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

from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.particles.base.baseparticle import BaseParticle
from edelweissmpm.sets.particleset import ParticleSet


def generateCooksMembraneParticleGrid(
    model: MPMModel,
    journal: Journal,
    particleFactoryCallback: typing.Callable[[int, np.ndarray, float], BaseParticle],
    name: str = "cooks_membrane",
    x0: float = 0.0,
    y0: float = 0.0,
    l: float = 48.0,
    h0: float = 44.0,
    h1: float = 16.0,
    nX: int = 10,
    nY: int = 10,
    firstParticleNumber: int = 1,
    thickness: float = 1.0,
):
    """Generate a Cook's membrane-shaped grid of particles (trapezoidal mesh).

    Returns
    -------
    MPMModel
        The updated model.
    """

    x_coords = np.linspace(x0, x0 + l, nX + 1)
    vertices = np.zeros((nX + 1, nY + 1, 2))  # (x, y) coordinates
    dx = l / nX

    for i, x in enumerate(x_coords):
        y_top = y0 + h0 + h1 / l * (x - x0)
        y_bottom = y0 + h0 / l * (x - x0)

        y_coords = np.linspace(y_bottom, y_top, nY + 1)

        for j, y in enumerate(y_coords):
            vertices[i, j] = [x, y]

    # Create particles
    currentParticleNumber = firstParticleNumber
    particles = []

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
            quad_h1 = vertices[i, j + 1][1] - vertices[i, j][1]
            quad_h2 = vertices[i + 1, j + 1][1] - vertices[i + 1, j][1]
            pVolume = 0.5 * (quad_h1 + quad_h2) * dx * thickness
            particle = particleFactoryCallback(currentParticleNumber, quad, pVolume)
            model.particles[currentParticleNumber] = particle
            particles.append(particle)
            currentParticleNumber += 1

    particleGrid = np.asarray(particles).reshape(nX, nY)

    # Create sets (similar to before)
    model.particleSets[f"{name}_all"] = ParticleSet(f"{name}_all", particleGrid.flatten())
    model.particleSets[f"{name}_left"] = ParticleSet(f"{name}_left", list(particleGrid[0, :]))
    model.particleSets[f"{name}_right"] = ParticleSet(f"{name}_right", list(particleGrid[-1, :]))
    model.particleSets[f"{name}_top"] = ParticleSet(f"{name}_top", list(particleGrid[:, -1]))
    model.particleSets[f"{name}_bottom"] = ParticleSet(f"{name}_bottom", list(particleGrid[:, 0]))

    model.particleSets[f"{name}_leftBottom"] = ParticleSet(f"{name}_leftBottom", [particleGrid[0, 0]])
    model.particleSets[f"{name}_leftTop"] = ParticleSet(f"{name}_leftTop", [particleGrid[0, -1]])
    model.particleSets[f"{name}_rightBottom"] = ParticleSet(f"{name}_rightBottom", [particleGrid[-1, 0]])
    model.particleSets[f"{name}_rightTop"] = ParticleSet(f"{name}_rightTop", [particleGrid[-1, -1]])

    model.particleSets[f"{name}_boundary"] = ParticleSet(
        f"{name}_boundary",
        list(particleGrid[:, 0])
        + list(particleGrid[:, -1])
        + list(particleGrid[0, 1:-1])
        + list(particleGrid[-1, 1:-1]),
    )

    model.surfaces[f"{name}_bottom"] = {1: np.ravel(particleGrid[:, 0])}
    model.surfaces[f"{name}_top"] = {3: np.ravel(particleGrid[:, -1])}
    model.surfaces[f"{name}_right"] = {2: np.ravel(particleGrid[-1, :])}
    model.surfaces[f"{name}_left"] = {4: np.ravel(particleGrid[0, :])}

    return model
