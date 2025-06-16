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
from edelweissfe.journal.journal import Journal

from edelweissmpm.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmpm.particlemanagers.base.baseparticlemanager import BaseParticleManager


class _KDBinOrganizer:
    """A class to manage the kernel functions in cartesian bins in multiple dimension for fast access.

    We store the bounding box of each kerne lfunction (support) in the respective (partially-)covereing bins.
    This way, we can quickly find for given coordinates (of particles) the shape function candidates of which the support might cover the coordinates.

    Parameters
    ----------
    kernelFunctions
        The list of shape functions.
    dimension
        The dimension of the problem.
    """

    def __init__(self, kernelFunctions, dimension):
        self._dimension = dimension
        self._kernelFunctions = kernelFunctions

        # we make the bin size the average of the bounding box of the shape functions

        # boundingBoxes = np.array([sf.getBoundingBox() for sf in kernelFunctions])
        boundingBoxes = [sf.getBoundingBox() for sf in kernelFunctions]
        boundingBoxesMins = np.array([bb[0] for bb in boundingBoxes])
        boundingBoxesMaxs = np.array([bb[1] for bb in boundingBoxes])

        self._binSize = np.mean(boundingBoxesMaxs - boundingBoxesMins, axis=0) / 2
        self._boundingBoxMin = np.min(boundingBoxesMins, axis=0) - 1e-12
        self._boundingBoxMax = np.max(boundingBoxesMaxs, axis=0) + 1e-12

        self._nBins = np.ceil((self._boundingBoxMax - self._boundingBoxMin) / self._binSize).astype(int)

        self._thebins = np.frompyfunc(list, 0, 1)(np.empty(self._nBins, dtype=object))
        # print(self._thebins.shape)

        # for every shape function, we need to get the bin indices for the min. and max. bounding box:
        for i, kf in enumerate(kernelFunctions):
            minIndices = self._getBinIndices(boundingBoxes[i][0])
            maxIndices = self._getBinIndices(boundingBoxes[i][1])

            # now we need to loop over all bins in the bounding box of the shape function and add the shape function to the bin
            # 3d case:
            if self._dimension == 3:
                for i in range(minIndices[0], maxIndices[0] + 1):
                    for j in range(minIndices[1], maxIndices[1] + 1):
                        for k in range(minIndices[2], maxIndices[2] + 1):
                            self._thebins[i, j, k].append(kf)
            # 2d case:
            elif self._dimension == 2:
                for i in range(minIndices[0], maxIndices[0] + 1):
                    for j in range(minIndices[1], maxIndices[1] + 1):
                        self._thebins[i, j].append(kf)
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

        return [int((coordinate[i] - self._boundingBoxMin[i]) / self._binSize[i]) for i in range(self._dimension)]

    def getKernelFunctionCandidates(self, coordinate: np.ndarray) -> list:
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
        thebin = self._thebins[tuple(indices)]
        return thebin

    def __str__(self):
        return f"KDBin with {len(self._kernelFunctions)} shape functions in {self._nBins} bins of size {self._binSize} in a bounding box from {self._boundingBoxMin} to {self._boundingBoxMax}."


class KDBinOrganizedParticleManager(BaseParticleManager):
    """A k-dimensional bin organized manager for particles and meshfree shape functions  for locating points in supports.

    Parameters
    ----------
    meshfreeKernelFunctions
        The list of shape functions.
    particles
        The list of particles.
    dimension
        The dimension of the problem.
    """

    def __init__(
        self,
        particleKernelDomain: ParticleKernelDomain,
        dimension: int,
        journal: Journal,
        bondParticlesToKernelFunctions: bool = False,
    ):

        self._meshfreeKernelFunctions = particleKernelDomain.meshfreeKernelFunctions
        self._particles = particleKernelDomain.particles
        self._dimension = dimension
        self._bondParticlesToKernelFunctions = bondParticlesToKernelFunctions
        self._journal = journal

        if self._bondParticlesToKernelFunctions:
            if len(self._particles) != len(self._meshfreeKernelFunctions):
                raise ValueError("The number of particles and kernel functions must be equal.")
            # for particle, kernelFunction in zip(self._particles, self._meshfreeKernelFunctions):
            #     if not np.isclose(particle.getCenterCoordinates(), kernelFunction.center).all():
            #         raise ValueError(
            #             f"The particle and kernel function coordinates must be close to each other. {particle.getCenterCoordinates()} != {kernelFunction.center}"
            #         )

        self.signalizeKernelFunctionUpdate()

    def signalizeKernelFunctionUpdate(
        self,
    ):
        self._theBins = _KDBinOrganizer(self._meshfreeKernelFunctions, self._dimension)

    def updateConnectivity(
        self,
    ):
        hasChanged = False

        if self._bondParticlesToKernelFunctions:
            self._journal.message(
                f"Updating kernel function positions for the bonding definition with {len(self._particles)} particles.",
                "ParticleManager",
            )
            for particle, kernelFunction in zip(self._particles, self._meshfreeKernelFunctions):
                kernelFunction.moveTo(particle.getCenterCoordinates())

            self.signalizeKernelFunctionUpdate()

        for p in self._particles:
            evaluationCoordinates = p.getEvaluationCoordinates()

            # rough search based on the bounding box of the kernel functions
            kernelFunctionCandidates = {
                candidate
                for vertex in evaluationCoordinates
                for candidate in self._theBins.getKernelFunctionCandidates(vertex)
            }
            # fine search based on the actual support of the kernel functions
            kernelFunctions = {
                sf
                for coordinate in evaluationCoordinates
                for sf in kernelFunctionCandidates
                if sf.isCoordinateInCurrentSupport(coordinate)
            }

            # we need to sort the kernel functions using their node number to ensure that the order is always the same
            kernelFunctions = list(sorted(kernelFunctions, key=lambda x: x.node.label))

            if hasChanged or kernelFunctions != p.kernelFunctions:
                hasChanged = True

            p.assignKernelFunctions(kernelFunctions)

        return hasChanged

    def getCoveredDomain(
        self,
    ):
        return self._theBins._boundingBoxMin, self._theBins._boundingBoxMax

    def __str__(self):
        return f"KDBinOrganizedParticleManager with {len(self._particles)} particles and {len(self._meshfreeKernelFunctions)} shape functions in {self._dimension} dimensions. Covered domain: {self.getCoveredDomain()}."

    def visualize(self):
        """For 2D only: Visualize the number of kernel functions in the bins."""

        if self._dimension != 2:
            raise ValueError("Visualization only supported for 2D.")

        import matplotlib.pyplot as plt

        nBins = self._theBins._nBins
        nKernelFunctions = np.zeros(nBins)
        for i in range(nBins[0]):
            for j in range(nBins[1]):
                nKernelFunctions[i, j] = len(self._theBins._thebins[i, j])

        plt.figure()
        plt.imshow(
            nKernelFunctions.T,
        )
        plt.title("Number of kernel functions in the bins of the KDBinOrganizer")

        nBins = self._theBins._nBins
        # plot the lines:
        for i in range(nBins[0] + 1):
            plt.plot([i - 0.5, i - 0.5], [0 - 0.5, nBins[1] - 0.5], "k")
        for j in range(nBins[1] + 1):
            plt.plot([0 - 0.5, nBins[0] - 0.5], [j - 0.5, j - 0.5], "k")

        plt.colorbar()
        plt.show()
