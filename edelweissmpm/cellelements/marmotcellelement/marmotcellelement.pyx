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
cimport numpy as np
cimport edelweissmpm.cellelements.marmotcellelement.marmotcellelement
cimport libcpp.cast
cimport cython

from edelweissfe.utils.exceptions import CutbackRequest
from libcpp.memory cimport unique_ptr, allocator, make_unique
from libc.stdlib cimport malloc, free

from edelweissmpm.cells.marmotcell.marmotcell cimport MarmotCellWrapper, MarmotCell
from edelweissmpm.materialpoints.marmotmaterialpoint.mp cimport MarmotMaterialPointWrapper
    
@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotCellElementWrapper(MarmotCellWrapper):
    """This cell as a wrapper for MarmotCellElements.

    For the documentation of MarmotCellElements, please refer to `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.

    Parameters
    ----------
    cellElementType
        The Marmot element which should be represented, e.g., CPE4.
    cellElementNumber
        The (unique) label of this CellElement.
    gridnodes
        The list of gridnodes for this CellElement.
        """
    def __init__(self, cellType, cellNumber, gridnodes):
        super().__init__(cellType, cellNumber, gridnodes)

    @property
    def nMaterialPoints(self):
        return self._nMaterialPoints

    def getRequestedMaterialPointCoordinates(self, ):
        cdef int dim = self._gridnodeCoordinates.shape[1]
        cdef np.ndarray mpCoordinates = np.zeros( (self._nMaterialPoints, dim) )
        cdef double[:,::1] _coordsView = mpCoordinates
        self._marmotCellElement.getRequestedMaterialPointCoordinates(&_coordsView[0, 0])
        return mpCoordinates

    def getRequestedMaterialPointVolumes(self, ):
        cdef np.ndarray vols = np.zeros(self._nMaterialPoints)
        cdef double[::1] _volsView = vols
        self._marmotCellElement.getRequestedMaterialPointVolumes(&_volsView[0])
        return vols

