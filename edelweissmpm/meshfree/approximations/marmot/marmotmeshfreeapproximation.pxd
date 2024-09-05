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

cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np

from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction cimport (
    MarmotMeshfreeKernelFunction,
)


cdef extern from "Marmot/MarmotMeshfreeApproximation.h" namespace "Marmot::Meshfree":
    cdef cppclass MarmotMeshfreeApproximation nogil:
        void computeShapeFunctions(const double* coord,
                                   const vector[ const MarmotMeshfreeKernelFunction* ]& kernelFunctions,
                                   double* shapeFunctionValues ) const

        void computeShapeFunctionsAndGradients(const double* coord,
                                               const vector[ const MarmotMeshfreeKernelFunction* ]& kernelFunctions,
                                               double* shapeFunctionValues,
                                               double* shapeFunctionGradients ) const

cdef extern from "Marmot/MarmotMeshfreeReproducingKernelApproximation.h" namespace "Marmot::Meshfree":
    cdef cppclass MarmotMeshfreeReproducingKernelApproximation nogil:
        MarmotMeshfreeReproducingKernelApproximation(int dim, int completenessOrder)


cdef class MarmotMeshfreeApproximationWrapper:

    cdef int _nDim
    cdef MarmotMeshfreeApproximation* _marmotMeshfreeApproximation
