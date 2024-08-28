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

import math

import numpy as np

from edelweissmpm.meshfreeshapefunctions.base.basemeshfreeshapefunction import (
    BaseMeshfreeShapeFunction,
)
from edelweissmpm.particlemanagers.base.baseparticlemanager import BaseParticleManager
from edelweissmpm.particles.base.baseparticle import BaseParticle


class _KDBinOrganizer:
    """A class to manage the shape functions in cartesian bins in multiple dimension for fast access.

    We store the bounding box of each shape function (support) in the respective (partially-)covereing bins.
    This way, we can quickly find for given coordinates (of particles) the shape function candidates of which the support might cover the coordinates.

    Parameters
    ----------
    shapeFunctions
        The list of shape functions.
    dimension
        The dimension of the problem.
    """

    def __init__(self, shapeFunctions, dimension):
        self._dimension = dimension
        self._shapeFunctions = shapeFunctions

        # we make the bin size the average of the bounding box of the shape functions
        self._binSize = np.mean([sf.getBoundingBox()[1] - sf.getBoundingBox()[0] for sf in shapeFunctions], axis=0)

        self._boundingBoxMin = [min([sf.getBoundingBox()[0][i] for sf in shapeFunctions]) for i in range(dimension)]
        self._boundingBoxMax = [max([sf.getBoundingBox()[1][i] for sf in shapeFunctions]) for i in range(dimension)]

        self._nBins = (math.ceil(s) for s in (self._boundingBoxMax - self._boundingBoxMin) / self._binSize)
        # self._thebins = np.empty( self._nBins, dtype=list)

        self._thebins = np.frompyfunc(list, 0, 1)(np.empty(self._nBins, dtype=object))

        # for every shape function, we need to get the bin indices for the min. and max. bounding box:
        for sf in shapeFunctions:
            minIndices = self.getBinIndices(sf.getBoundingBox()[0])
            maxIndices = self.getBinIndices(sf.getBoundingBox()[1])

            # now we need to loop over all bins in the bounding box of the shape function and add the shape function to the bin
            # 3d case:
            if self._dimension == 3:
                for i in range(minIndices[0], maxIndices[0] + 1):
                    for j in range(minIndices[1], maxIndices[1] + 1):
                        for k in range(minIndices[2], maxIndices[2] + 1):
                            self._thebins[i, j, k].append(sf)
            # 2d case:
            elif self._dimension == 2:
                for i in range(minIndices[0], maxIndices[0] + 1):
                    for j in range(minIndices[1], maxIndices[1] + 1):
                        self._thebins[i, j].append(sf)
            # 1d case:
            elif self._dimension == 1:
                for i in range(minIndices[0], maxIndices[0] + 1):
                    self._thebins[i].append

            else:
                raise ValueError("Dimension not supported")

    def _getBinIndices(self, coordinate: np.ndarray) -> list:
        """Get the cartesian bin indices for a given coordinate.

        Parameters
        ----------
        coordinate
            The coordinate for which the indices should be computed.

        Returns
        -------
        list
            The list of indices.
        """

        return [int((coordinate[i] - self._boundingBox[0][i]) / self._binSize[i]) for i in range(self._dimension)]

    def getShapeFunctionCandidates(self, coordinate: np.ndarray) -> list:
        """Get the shape function candidates for a given coordinate.

        Candidates are shape functions that might cover the given coordinate.
        It is guaranteed that all candidates are returned, but not that all returned shape functions might cover the coordinate,
        since this is only a fast pre-selection based on the bounding box of the shape functions.

        Parameters
        ----------
        coordinate
            The coordinate for which the shape function candidates should be determined.

        Returns
        -------
        list
            The list of shape function candidates.
        """

        indices = self._getBinIndices(coordinate)
        return self._thebins[indices]

    def __str__(self):
        return f"KDBin with {len(self._shapeFunctions)} shape functions in {self._nBins} bins of size {self._binSize} in a bounding box from {self._boundingBoxMin} to {self._boundingBoxMax}."


class KDBinOrganizedParticleManager(BaseParticleManager):
    """A k-dimensional bin organized manager for particles and meshfree shape functions  for locating points in supports.

    Parameters
    ----------
    meshfreeShapeFunctions
        The list of shape functions.
    particles
        The list of particles.
    dimension
        The dimension of the problem.
    """

    def __init__(
        self,
        meshfreeShapeFunctions: list[BaseMeshfreeShapeFunction],
        particles: list[BaseParticle],
        dimension: int,
    ):

        self._meshfreeShapeFunctions = meshfreeShapeFunctions
        self._particles = particles
        self._dimension = dimension
        self.signalizeShapeFunctionLocationUpdate()

    def signalizeShapeFunctionLocationUpdate(
        self,
    ):
        self._theBins = _KDBinOrganizer(self._meshfreeShapeFunctions, self._dimension)

    def updateConnectivity(
        self,
    ):
        hasChanged = False
        for p in self._particles:
            shapeFunctionCandidates = self._theBins.getShapeFunctionCandidates(p.getCoordinates())

            shapeFunctions = set(
                sf
                for coordinate in p.getVertexCoordinates()
                for sf in shapeFunctionCandidates
                if sf.isInSupport(coordinate)
            )

            if shapeFunctions != p.getAssignedShapeFunctions():
                hasChanged = True
                p.assignShapeFunctions(shapeFunctions)

        return hasChanged
