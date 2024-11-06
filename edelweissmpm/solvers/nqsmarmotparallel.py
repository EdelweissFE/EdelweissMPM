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
import os
from multiprocessing import cpu_count

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix

from edelweissmpm.solvers.base.parallelization import (
    computeMarmotCellsInParallel,
    computeMarmotMaterialPointsInParallel,
    computeMarmotParticlesInParallel,
)
from edelweissmpm.solvers.nqs import NonlinearQuasistaticSolver


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

        if "OMP_NUM_THREADS" in os.environ:
            self.numThreads = int(os.environ["OMP_NUM_THREADS"])

        super().__init__(journal)

    @performancetiming.timeit("computation material points")
    def _computeMaterialPoints(self, materialPoints_, time: float, dT: float):
        return computeMarmotMaterialPointsInParallel(materialPoints_, time, dT, self.numThreads)

    @performancetiming.timeit("computation active cells")
    def _computeCells(
        self,
        activeCells_: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        return computeMarmotCellsInParallel(
            activeCells_,
            dU,
            P,
            F,
            K_VIJ,
            time,
            dT,
            theDofManager,
            self.numThreads,
        )

    @performancetiming.timeit("computation particles")
    def _computeParticles(
        self,
        particles_: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        return computeMarmotParticlesInParallel(
            particles_,
            dU,
            P,
            F,
            K_VIJ,
            time,
            dT,
            theDofManager,
            self.numThreads,
        )
