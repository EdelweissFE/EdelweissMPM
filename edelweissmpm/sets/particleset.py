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

from edelweissfe.sets.orderedset import ImmutableOrderedSet

from edelweissmpm.particles.base.baseparticle import BaseParticle
from edelweissmpm.particles.marmot.marmotparticlewrapper import MarmotParticleWrapper


class ParticleSet(ImmutableOrderedSet):
    """A basic particle set.
    It has a label, and a list containing the unique particles.

    Parameters
    ----------
    label
        The unique label for this element set.
    particles
        A list of particles.
    """

    def __init__(
        self,
        label: str,
        particles: list[BaseParticle],
    ):
        self.allowedObjectTypes = [BaseParticle, MarmotParticleWrapper]

        super().__init__(label, particles)
        self.particles = self.items
