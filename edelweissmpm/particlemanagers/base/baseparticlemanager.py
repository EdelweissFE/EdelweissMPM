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
Particle managers are responsible for the connectivity between particles and shape functions.
They track the particles and shape functions and update the connectivity between them.

Compared to the MPM, several subtle differences exist between the particle managers:
- Each particle vertice may be reside in multiple shape functions, whereas a material point vertex can be in only one cell.
- For nodally integrated ((semi-)Lagrangian) shape functions, no inactive cells exist, whereas for the material point method, inactive cells exist.
"""

from abc import ABC, abstractmethod


class BaseParticleManager(ABC):
    """
    The BaseParticleManager class is an abstract base class for all particle managers.
    If you want to implement a new particle manager, you have to inherit from this class."""

    @abstractmethod
    def updateConnectivity(
        self,
    ) -> bool:
        """
        Update the connectivity between observed shape functions and particles.
        Returns the truth value of a change during the last connectivity update.

        Returns
        -------
        bool
            The truth value of change.
        """

    @abstractmethod
    def signalizeKernelFunctionUpdate(
        self,
    ) -> None:
        """
        For semi-Lagrangian methods, the kernel function locations may change.
        This method is to signalize such a to the material point manager, so that it can account for the changed supports.
        """
