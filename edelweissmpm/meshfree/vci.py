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
    nDim
        The number of dimensions of the problem (e.g., 2D or 3D).
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
        self._nDim = particles[0].dimension

        # we need to know the number of VCI constraints. Since this concept only makes sense for a set of identical particle types,
        # we can just take the first particle
        # Most likely having a set of particles with different VCI constraints is not useful
        self._nVCIConstraints = particles[0].vci_getNumberOfConstraints()

    def computeVCICorrections(self):

        # each kernel function has its place in the global vector (similar to a dof vector)
        global_testFunction_indices = {kf: i for i, kf in enumerate(self._kernelFunctions)}

        nTestFunctions = len(global_testFunction_indices)

        # print("Kernel function indices: ", kfIndices)
        # sizeResiduals = nShape * self._nVCIConstraints * self._nDim # for integrals/residuals in total (global)

        # for computing the particles' contributions to the integrals, we employ a larger vector,
        # which contains the contributions of all particles for each kernel function
        size_gather_for_testFunctions = 0
        for p in self._particles:
            size_gather_for_testFunctions += len(p.kernelFunctions)

        # the relation between the scattered vector and the global vector is given by the indices array
        gather_TestFunction_indices_to_global = np.zeros(size_gather_for_testFunctions, dtype=int)

        particles_testFunction_blocks_in_gather = {}
        current_block_start = 0
        for p in self._particles:
            kf = p.kernelFunctions
            particles_testFunction_blocks_in_gather[p] = slice(current_block_start, current_block_start + len(kf))
            for k in kf:
                gather_TestFunction_indices_to_global[current_block_start] = global_testFunction_indices[k]
                current_block_start += 1

        for _ in range(1):

            # now we can compute the integrals
            gather_PsiGrad_P_Integral = np.zeros((size_gather_for_testFunctions, self._nDim, self._nVCIConstraints))
            gather_Psi_PGrad_Integral = np.zeros_like(gather_PsiGrad_P_Integral)
            gather_Psi_P_BoundaryIntegral = np.zeros_like(gather_PsiGrad_P_Integral)

            gather_mMatrices = np.zeros((size_gather_for_testFunctions, self._nVCIConstraints, self._nVCIConstraints))

            for p in self._particles:

                particle_testFunction_block_in_gather = particles_testFunction_blocks_in_gather[p]

                p.vci_compute_TestGradient_P_Integral(
                    gather_PsiGrad_P_Integral[particle_testFunction_block_in_gather, :, :]
                )  # this works due to the row-major order of the arrays, resulting the individual contributions to be stored in a contiguous block
                p.vci_compute_Test_PGradient_Integral(
                    gather_Psi_PGrad_Integral[particle_testFunction_block_in_gather, :, :]
                )
                p.vci_compute_MMatrix(gather_mMatrices[particle_testFunction_block_in_gather, :, :])

            for particleBoundaryDefinition in self._particleBoundaryDefinitions:
                for p in particleBoundaryDefinition.particles:

                    particle_testFunction_block_in_gather = particles_testFunction_blocks_in_gather[p]
                    p.vci_compute_Test_P_BoundaryIntegral(
                        gather_Psi_P_BoundaryIntegral[particle_testFunction_block_in_gather, :, :],
                        particleBoundaryDefinition.boundarySurfaceVector,
                        particleBoundaryDefinition.boundaryID,
                    )

            Psi_P_BoundaryIntegral = np.zeros((nTestFunctions, self._nDim, self._nVCIConstraints))
            PsiGrad_P_Integral = np.zeros_like(Psi_P_BoundaryIntegral)
            Psi_PGrad_Integral = np.zeros_like(Psi_P_BoundaryIntegral)
            mMatrices = np.zeros((nTestFunctions, self._nVCIConstraints, self._nVCIConstraints))

            # scatter the contributions to the global vector

            # np.add.at(gather_Psi_P_BoundaryIntegral, gather_TestFunction_indices_to_global, Psi_P_BoundaryIntegral)
            # np.add.at(gather_PsiGrad_P_Integral, gather_TestFunction_indices_to_global, PsiGrad_P_Integral)
            # np.add.at(gather_Psi_PGrad_Integral, gather_TestFunction_indices_to_global, Psi_PGrad_Integral)
            # np.add.at(gather_mMatrices, gather_TestFunction_indices_to_global, mMatrices)

            for i in range(size_gather_for_testFunctions):
                Psi_P_BoundaryIntegral[gather_TestFunction_indices_to_global[i], :, :] += gather_Psi_P_BoundaryIntegral[
                    i, :, :
                ]
                PsiGrad_P_Integral[gather_TestFunction_indices_to_global[i], :, :] += gather_PsiGrad_P_Integral[i, :, :]
                Psi_PGrad_Integral[gather_TestFunction_indices_to_global[i], :, :] += gather_Psi_PGrad_Integral[i, :, :]
                mMatrices[gather_TestFunction_indices_to_global[i], :, :] += gather_mMatrices[i, :, :]

            # print("Boundary integrals: ", boundaryIntegrals)
            # print("Volume integrals: ", volumeIntegrals)
            # print("Diff: ", volumeIntegrals - boundaryIntegrals)
            # print("Kernel localization integrals: ", kernelLocalizationIntegrals)

            # now we can compute the correction terms
            # residual = Psi_P_BoundaryIntegral - PsiGrad_P_Integral - Psi_PGrad_Integral
            # print(residual.reshape( (nTestFunctions, -1)))
            # residual = PsiGrad_P_Integral
            # print(residual.reshape( (nTestFunctions, -1)))
            # print max abs of residual
            # max_residual = np.max(np.abs(residual))
            # print("Max residual: ", max_residual)

            # print(mMatrices.flatten())

            if self._nVCIConstraints == 0:
                eta_AjC = np.einsum(
                    "ACD,AjD->AjC", np.reciprocal(mMatrices), (Psi_P_BoundaryIntegral - PsiGrad_P_Integral)
                )

            else:
                residuals = Psi_P_BoundaryIntegral - PsiGrad_P_Integral - Psi_PGrad_Integral
                eta_AjC = np.zeros_like(residuals)

                for A in range(nTestFunctions):
                    # eta_jC = M_A_CD^-1 * residuals_A_jC
                    eta_AjC[A, :, :] = np.linalg.solve(mMatrices[A, :, :], residuals[A, :, :].T).T

            for p in self._particles:
                particle_testFunction_block_in_gather = particles_testFunction_blocks_in_gather[p]
                p.vci_assignTestFunctionCorrectionTerms(
                    eta_AjC[gather_TestFunction_indices_to_global[particle_testFunction_block_in_gather], :, :]
                )

    @property
    def meshfreeKernelFunctions(self) -> list[BaseMeshfreeKernelFunction]:
        return self._kernelFunctions
