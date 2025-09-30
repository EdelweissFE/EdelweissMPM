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
""" """

from abc import ABC, abstractmethod

import numpy as np

from edelweissmpm.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)


class BaseMeshfreeApproximation(ABC):
    """Base class for meshfree approximations.

    In meshfree methods (MLS, RKPM), based on a finite number of kernel (RKPM) or weight (MLS) functions, an approximations can be computed at any point in the domain. This class provides the interface for the computation of such approximations.

    """

    @abstractmethod
    def computeShapeFunctionValues(
        self, coordinates: np.ndarray, kernelfunctions: list[BaseMeshfreeKernelFunction]
    ) -> np.ndarray:
        """Compute the shape function values at the given coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            The coordinates at which the shape functions should be evaluated.
        kernelfunctions : list[BaseKernelFunction]
            The kernel functions that should be used to compute the shape functions.

        Returns
        -------
        np.ndarray
            The shape function values at the given coordinates.
        """
