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
Variationally Consistent Integration is a concept by Chen, Hillman and Ruter (2013) to ensure Galerkin exactness of meshfree methods
by employing a Petrov-Galerkin formulation.

It requires the computation of a set of correction terms, depending on the desired order of Galerkin exactness.

The implementation, albeit trivial in concept, does not readily fit into the existing workflow of nonlinear simulations:
Before computing the weak form, the correction terms need to be computed, and the weak form needs to be adjusted accordingly.
The correction cannot be computed locally (i.e., for each particle during the weak form computation), as it requires
global volume and boundary integrals.

"""

import numpy as np

from edelweissmpm.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)
from edelweissmpm.particles.base.baseparticle import BaseParticle


class BoundaryParticleDefinition:

    def __init__(
        self, particles: list[BaseParticle], boundarySurfaceVector: np.ndarray, particleBoundaryFaceID: int = -1
    ):
        self.particles = particles
        self.boundarySurfaceVector = boundarySurfaceVector
        self.boundaryID = particleBoundaryFaceID


class VariationallyConsistentIntegrationManager:
    """A class to describe the potential interaction between a set of particles and a set of kernel kernel functions.

    Parameters
    ----------
    particles
        The list of particles.
    kernelFunctions
        The list of kernel functions.
    particleBoundaryDefinitions
        The list of particle boundary definitions used for the boundary integrals.
    """

    def __init__(
        self,
        particles: list[BaseParticle],
        meshfreeKernelFunctions: list[BaseMeshfreeKernelFunction],
        particleBoundaryDefinitions: list[BoundaryParticleDefinition],
    ):
        # TODO: replace with ParticleKernelDomain
        self._particles = particles
        self._kernelFunctions = meshfreeKernelFunctions
        self._particleBoundaryDefinitions = particleBoundaryDefinitions

        # we need to know the number of VCI constraints. Since this concept only makes sense for a set of identical particle types,
        # we can just take the first particle
        #
        # TODO: We may change this in future, but most likely having a set of particles with different VCI constraints is not useful
        self._nVCIConstraints = particles[0].getNumberOfVCIConstraints()

    def computeVCICorrections(self):

        # each kernel function has its place in the global vector (similar to a dof vector)
        kfIndices = {kf: i for i, kf in enumerate(self._kernelFunctions)}
        # print("Kernel function indices: ", kfIndices)
        sizeGlobalVec = len(kfIndices)

        # for computing the particles' contributions to the integrals, we employ a larger vector,
        # which contains the contributions of all particles
        sizeGatherVec = 0
        for p in self._particles:
            sizeGatherVec += len(p.kernelFunctions)

        # the relation between the scattered vector and the global vector is given by the indices array
        indices = np.zeros(sizeGatherVec, dtype=int)
        particlesGatherLocation = {}
        currentOffset = 0
        for p in self._particles:
            kf = p.kernelFunctions

            sizeParticle = len(kf)
            particlesGatherLocation[p] = slice(currentOffset, currentOffset + sizeParticle)
            for k in kf:
                indices[currentOffset] = kfIndices[k]
                currentOffset += 1

        for constraint in range(self._nVCIConstraints):

            # now we can compute the integrals
            particleBoundaryIntegrals = np.zeros(sizeGatherVec)
            particleVolumeIntegrals = np.zeros(sizeGatherVec)
            particleKernelLocalizationIntegrals = np.zeros(sizeGatherVec)

            for p in self._particles:

                loc = particlesGatherLocation[p]

                p.computeTestFuntionGradientVolumeIntegral(particleVolumeIntegrals[loc], constraint)
                p.computeKernelLocalizationIntegral(particleKernelLocalizationIntegrals[loc], constraint)

            for particleBoundaryDefinition in self._particleBoundaryDefinitions:
                for p in particleBoundaryDefinition.particles:

                    loc = particlesGatherLocation[p]
                    p.computeTestFunctionBoundaryIntegral(
                        particleBoundaryIntegrals[loc],
                        particleBoundaryDefinition.boundarySurfaceVector,
                        particleBoundaryDefinition.boundaryID,
                        constraint,
                    )

            # now we need to scatter the contributions to the global vector
            boundaryIntegrals = np.zeros(sizeGlobalVec)
            volumeIntegrals = np.zeros(sizeGlobalVec)
            kernelLocalizationIntegrals = np.zeros(sizeGlobalVec)

            np.add.at(boundaryIntegrals, indices, particleBoundaryIntegrals)
            np.add.at(volumeIntegrals, indices, particleVolumeIntegrals)
            np.add.at(kernelLocalizationIntegrals, indices, particleKernelLocalizationIntegrals)

            # print("Boundary integrals: ", boundaryIntegrals)
            # print("Volume integrals: ", volumeIntegrals)
            # print("Diff: ", volumeIntegrals - boundaryIntegrals)
            # print("Kernel localization integrals: ", kernelLocalizationIntegrals)

            # now we can compute the correction terms
            correctionTerms = -np.reciprocal(kernelLocalizationIntegrals) * (volumeIntegrals - boundaryIntegrals)

            # now we need to scatter the correction terms to the particles
            for p in self._particles:
                loc = particlesGatherLocation[p]
                p.assignShapeFunctionCorrectionTerm(correctionTerms[indices[loc]], constraint)

    @property
    def meshfreeKernelFunctions(self) -> list[BaseMeshfreeKernelFunction]:
        return self._kernelFunctions
