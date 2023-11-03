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

from mpm.materialpoints.base.mp import BaseMaterialPoint
import numpy as np


class MaterialPoint(BaseMaterialPoint):
    shape = "point"

    def __init__(self, formulation: str, label: int, coordinates: np.ndarray, volume: float):
        self._label = label
        self._coordinates = coordinates
        self._volume = volume

        self._displacement = np.zeros(2)

    @property
    def label(self) -> int:
        return self._label

    @property
    def ensightType(self) -> str:
        return self.shape

    def getVertexCoordinates(
        self,
    ) -> np.ndarray:
        return np.reshape(self._coordinates + self._displacement, (1, 2))

    def getCenterCoordinates(
        self,
    ) -> np.ndarray:
        return self._coordinates + self._displacement

    def getVolume(
        self,
    ) -> float:
        return self._volume

    # def computeMaterialResponse(self, timeStep: float, timeTotal: float, dT: float):
    #     pass

    def acceptLastState(
        self,
    ):
        pass

    def resetToLastValidState(
        self,
    ):
        pass

    def getResultArray(self, result: str, getPersistentView: bool = True) -> np.ndarray:
        if result == "displacement":
            return self._displacement

    def setProperties(self, propertyName: str, elementProperties: np.ndarray):
        pass

    def initializeMaterialPoint(
        self,
    ):
        pass

    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        pass

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        pass

    def addDisplacement(self, dU: np.ndarray):
        self._displacement += dU

    def computeMaterialResponse(self, timeStep, timeTotal, dT):
        pass
