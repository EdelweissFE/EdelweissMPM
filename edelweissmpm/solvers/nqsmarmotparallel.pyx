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
import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
from edelweissfe.journal.journal import Journal

from edelweissmpm.solvers.nqs import NonlinearQuasistaticSolver

from cython.parallel cimport parallel, prange, threadid
from libc.stdlib cimport free, malloc
from libcpp.string cimport string

from edelweissmpm.cells.marmotcell.marmotcell cimport MarmotCell, MarmotCellWrapper
from edelweissmpm.materialpoints.marmotmaterialpoint.mp cimport (
    MarmotMaterialPoint,
    MarmotMaterialPointWrapper,
)
from edelweissmpm.particles.marmot.marmotparticlewrapper cimport (
    MarmotParticle,
    MarmotParticleWrapper,
)

import os
from multiprocessing import cpu_count
from time import time as getCurrentTime


class NQSParallelForMarmot(NonlinearQuasistaticSolver):
    """This is a parallel implemenntation of the NonlinearQuasistaticSolver.
    It only works with MarmotCells and MarmotElements, as it directly accesses and exploits
    the background Marmot C++ objects.

    It uses Cython/OpenMP for evaluation those MarmotCells and MarmotMaterialPoints in a prange loop,
    allowing to bypass the GIL and get decent performance.

    The number of threads for the OpenMP loop is determined based on the cpu count,
    or (higher priority) based on the environment variable OMP_NUM_THREADS.

    Parameters
    ----------
    journal
        The Journal instance for loggin purposes.
    """

    identification = "NQSParallelForMarmot"

    def __init__(self, journal: Journal):
        self.numThreads = cpu_count()

        if 'OMP_NUM_THREADS' in os.environ:
            self.numThreads = int( os.environ ['OMP_NUM_THREADS'] )

        super().__init__(journal)

    @performancetiming.timeit("computation material points")
    def _computeMaterialPoints(self, materialPoints_, float time, float dT):
        """Evaluate all material points' physics in an OpenMP prange loop.

        Parameters
        ----------
        materialPoints
            The list material points to  evaluated.
        time
            The current time.
        dT
            The increment of time.
        """

        cdef:
            list materialPoints = list(materialPoints_)
            int desiredThreads = self.numThreads
            int nMPs = len(materialPoints)
            int i

            MarmotMaterialPointWrapper backendBasedCythonMaterialPoint
            MarmotMaterialPoint** cppMPs = <MarmotMaterialPoint**> malloc ( nMPs * sizeof(MarmotMaterialPoint*) )

        for i in range(nMPs):
            backendBasedCythonMaterialPoint = materialPoints[i]
            cppMPs[i]                       = backendBasedCythonMaterialPoint._marmotMaterialPoint

        try:
            for i in prange(nMPs,
                        schedule='dynamic',
                        num_threads=desiredThreads,
                        nogil=True):

                (<MarmotMaterialPoint*> cppMPs[i] )[0].computeYourself(time, dT,)

        finally:
            free( cppMPs )

    @performancetiming.timeit("computation active cells")
    def _computeCells(
        self,
        activeCells_,
        double[::1] dU,
        double[::1] P,
        double[::1] F,
        K_VIJ,
        float time,
        float dT,
        theDofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        activeCells
            The list of (active) cells to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        cdef:
            int cellNDof, cellNumber, cellIdxInVIJ, cellIdxInPe, threadID, currentIdxInU
            int desiredThreads = self.numThreads
            int nActiveCells = len(activeCells_)
            list activeCells = list(activeCells_)

            long[::1] I             = K_VIJ.I
            double[::1] K_mView     = K_VIJ
            double[::1] dU_mView    = dU
            double[::1] P_mView     = P
            double[::1] F_mView     = F

            # oversized Buffers for parallel computing:
            # tables [nThreads x max(activeCells.ndof) ] for U & dU (can be overwritten during parallel computing)
            maxNDofOfAnyCell      = theDofManager.largestNumberOfCellNDof
            double[:, ::1] dUe  = np.empty((desiredThreads, maxNDofOfAnyCell), )
            # oversized buffer for Pe ( size = sum(activeCells.ndof) )
            double[::1] Pe = np.zeros(theDofManager.accumulatedCellNDof)


            MarmotCellWrapper backendBasedCythonCell
            # lists (cpp activeCells + indices and nDofs), which can be accessed parallely
            MarmotCell** cppActiveCells =      <MarmotCell**> malloc ( nActiveCells * sizeof(MarmotCell*) )
            int[::1] cellIndicesInVIJ         = np.empty( (nActiveCells,), dtype=np.intc )
            int[::1] cellIndexInPe            = np.empty( (nActiveCells,), dtype=np.intc )
            int[::1] cellNDofs                = np.empty( (nActiveCells,), dtype=np.intc )

            int i,j=0

        for i in range(nActiveCells):
            # prepare all lists for upcoming parallel element computing
            backendBasedCythonCell   = activeCells[i]
            cppActiveCells[i]              = backendBasedCythonCell._marmotCell
            cellIndicesInVIJ[i]           = theDofManager.idcsOfHigherOrderEntitiesInVIJ[backendBasedCythonCell]
            cellNDofs[i]                  = backendBasedCythonCell.nDof
            # each element gets its place in the Pe buffer
            cellIndexInPe[i] = j
            j += cellNDofs[i]

        try:
            for i in prange(nActiveCells,
                        schedule='dynamic',
                        num_threads=desiredThreads,
                        nogil=True):

                threadID      = threadid()
                cellIdxInVIJ  = cellIndicesInVIJ[i]
                cellIdxInPe   = cellIndexInPe[i]
                cellNDof = cellNDofs[i]

                for j in range (cellNDof):
                    # copy global U & dU to buffer memories for element eval.
                    currentIdxInU =     I [ cellIndicesInVIJ[i] +  j ]
                    dUe[threadID, j] =  dU_mView[ currentIdxInU ]

                (<MarmotCell*>
                     cppActiveCells[i] )[0].computeMaterialPointKernels(
                                                        &dUe[threadID, 0],
                                                        &Pe[cellIdxInPe],
                                                        &K_mView[cellIdxInVIJ],
                                                        time,
                                                        dT)

            #successful activeCells evaluation: condense oversize Pe buffer -> P
            for i in range(nActiveCells):
                cellIdxInVIJ =    cellIndicesInVIJ[i]
                cellIdxInPe =     cellIndexInPe[i]
                cellNDof =   cellNDofs[i]
                for j in range (cellNDof):
                    P_mView[ I[cellIdxInVIJ + j] ] +=      Pe[ cellIdxInPe + j ]
                    F_mView[ I[cellIdxInVIJ + j] ] += abs( Pe[ cellIdxInPe + j ] )
        finally:
            free( cppActiveCells )

    @performancetiming.timeit("computation particles")
    def _computeParticles(
        self,
        particles_,
        double[::1] dU,
        double[::1] P,
        double[::1] F,
        K_VIJ,
        float time,
        float dT,
        theDofManager,
    ):
        """Evaluate all particles.

        Parameters
        ----------
        particles
            The list of particles to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        cdef:
            int particleNDof, particleNumber, particleIdxInVIJ, particleIdxInPe, threadID, currentIdxInU
            int desiredThreads = self.numThreads
            int nActiveParticles = len(particles_)
            list particles = list(particles_)

            long[::1] I             = K_VIJ.I
            double[::1] K_mView     = K_VIJ
            double[::1] dU_mView    = dU
            double[::1] P_mView     = P
            double[::1] F_mView     = F

            # oversized Buffers for parallel computing:
            # tables [nThreads x max(particles.ndof) ] for U & dU (can be overwritten during parallel computing)
            maxNDofOfAnyParticle      = theDofManager.largestNumberOfParticleNDof
            double[:, ::1] dUe  = np.empty((desiredThreads, maxNDofOfAnyParticle), )
            # oversized buffer for Pe ( size = sum(particles.ndof) )
            double[::1] Pe = np.zeros(theDofManager.accumulatedParticleNDof)


            MarmotParticleWrapper backendBasedCythonParticle
            # lists (cpp particles + indices and nDofs), which can be accessed parallely
            MarmotParticle** cppActiveParticles =      <MarmotParticle**> malloc ( nActiveParticles * sizeof(MarmotParticle*) )
            int[::1] particleIndicesInVIJ         = np.empty( (nActiveParticles,), dtype=np.intc )
            int[::1] particleIndexInPe            = np.empty( (nActiveParticles,), dtype=np.intc )
            int[::1] particleNDofs                = np.empty( (nActiveParticles,), dtype=np.intc )

            int i,j=0

        for i in range(nActiveParticles):
            # prepare all lists for upcoming parallel element computing
            backendBasedCythonParticle   = particles[i]
            backendBasedCythonParticle._initializeStateVarsTemp()
            cppActiveParticles[i]             = backendBasedCythonParticle._marmotParticle
            particleIndicesInVIJ[i]           = theDofManager.idcsOfHigherOrderEntitiesInVIJ[backendBasedCythonParticle]
            particleNDofs[i]                  = backendBasedCythonParticle.nDof
            # each element gets its place in the Pe buffer
            particleIndexInPe[i] = j
            j += particleNDofs[i]

        try:
            for i in prange(nActiveParticles,
                        schedule='dynamic',
                        num_threads=desiredThreads,
                        nogil=True):


                threadID      = threadid()
                particleIdxInVIJ  = particleIndicesInVIJ[i]
                particleIdxInPe   = particleIndexInPe[i]
                particleNDof = particleNDofs[i]

                for j in range (particleNDof):
                    # copy global U & dU to buffer memories for element eval.
                    currentIdxInU =     I [ particleIndicesInVIJ[i] +  j ]
                    dUe[threadID, j] =  dU_mView[ currentIdxInU ]

                (<MarmotParticle*>
                     cppActiveParticles[i] )[0].computePhysicsKernels(
                                                        &dUe[threadID, 0],
                                                        &Pe[particleIdxInPe],
                                                        &K_mView[particleIdxInVIJ],
                                                        time,
                                                        dT)

            #successful particles evaluation: condense oversize Pe buffer -> P
            for i in range(nActiveParticles):
                particleIdxInVIJ =    particleIndicesInVIJ[i]
                particleIdxInPe =     particleIndexInPe[i]
                particleNDof =   particleNDofs[i]
                for j in range (particleNDof):
                    P_mView[ I[particleIdxInVIJ + j] ] +=      Pe[ particleIdxInPe + j ]
                    F_mView[ I[particleIdxInVIJ + j] ] += abs( Pe[ particleIdxInPe + j ] )
        finally:
            free( cppActiveParticles )

