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

import numpy as np

from libc.stdlib cimport free, malloc


cdef class MaterialPointResultCollector:

    cdef public  resultsTable

    cdef int nMPs, nSize
    cdef double[:, ::1] res_
    cdef double** resultPointers

    def __init__(self, materialPoints:list, result:str):
        """
        A cdef class for collecting materialPoint results (by using the permanent results pointer (i.e., a numpy array)
        in large array of all materialPoints and all quadrature points.

        Collecting materialPointal results may be a performance critical part.
        This cdef class allows for the efficient gathering.
        A 3D array is assembled if multiple quadrature points are requested (shape ``[materialPoints, quadraturePoints, resultVector]`` )
        or a 2D array for one quadrature point ( shape ``[materialPoints, resultVector]`` ).

        Method :func:`~edelweissfe.utils.materialPointresultcollector.MaterialPointResultCollector.getCurrentResults` updates the assembly array and passes it back.

        The caller is responsible to make a copy of it, if persistent results are needed!

        Parameters
        ----------
        materialPoints
            The list of materialPoints for which the results should be collected.
        result
            The name of the requested result.
        """

        # hotfix for cython compile error associated with 'range' typing of argument quadraturePoints
        # this is due to a bug in in cython 0.29.xx and should be fixed in cython 3.x.x
        # https://github.com/cython/cython/issues/4002

        self.nMPs = len(materialPoints)

        # assemble a 2d list of all permanent result arrays (=continously updated np arrays)
        resultsPointerList = [  el.getResultArray(result, getPersistentView=True)  for el in materialPoints ]
        self.nSize = resultsPointerList[0].shape[0]


        # allocate an equivalent 2D C-array for the pointers to each materialPoints results
        self.resultPointers = <double**> malloc ( sizeof(double*) * self.nMPs )

        cdef double* ptr
        cdef double[::1] res
        # fill the 2D C-array of pointers by accessing the materialPoints' resultArrays memoryviews
        for i, el in enumerate(resultsPointerList):
            # for j, gPt in enumerate(el):
            res = el
            ptr = <double*> &res[0]
            self.resultPointers[ i  ] = ptr

        # initialize the large assembly array
        self.resultsTable = np.empty([self.nMPs, self.nSize]  )

        # an internal use only memoryview is created for accesing the assembly
        self.res_ = self.resultsTable

    def update(self, ):
        """Update all results."""

        cdef int i, j, k
        for i in range(self.nMPs):
                for k in range(self.nSize):
                    # most inner loop: could also be handled by copying the complete vector at once,
                    # but this version turned out to be faster!
                    self.res_[i,k] = self.resultPointers[i][k]

    def getCurrentResults(self,) -> np.ndarray:
        """Update and get current results.

        Returns
        -------
        np.ndarray
            The results array.
        """

        self.update()
        return self.resultsTable

    def __dealloc__(self):
        free ( self.resultPointers )

