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
from fe.journal.journal import Journal
import fe.utils.performancetiming as performancetiming
from mpm.solvers.nqs import NonlinearQuasistaticSolver

import numpy as np
from cython.parallel cimport parallel, threadid, prange
from mpm.materialpoints.marmotmaterialpoint.mp cimport MarmotMaterialPointWrapper, MarmotMaterialPoint
from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from time import time as getCurrentTime
from multiprocessing import cpu_count
import os

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

    @performancetiming.timeit("solve step", "newton iteration", "computation material points")
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
