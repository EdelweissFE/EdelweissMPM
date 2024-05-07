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

cimport cython
cimport libcpp.cast
cimport numpy as np

cimport edelweissmpm.cellelements.marmotcellelement.marmotcellelement

from edelweissfe.utils.exceptions import CutbackRequest

from libc.stdlib cimport free, malloc
from libcpp.memory cimport allocator, make_unique, unique_ptr

from edelweissmpm.cells.marmotcell.marmotcell cimport (MarmotCell,
                                                       MarmotCellWrapper)
from edelweissmpm.materialpoints.marmotmaterialpoint.mp cimport \
    MarmotMaterialPointWrapper

from edelweissmpm.materialpoints.marmotmaterialpoint.mp import \
    MarmotMaterialPointWrapper


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotCellElementWrapper(MarmotCellWrapper):
    """This cell as a wrapper for MarmotCellElements.
    This class is not intended to be used directly, but to be subclassed by the actual cell element classes, such as LagrangianMarkerCellElement.

    For the documentation of MarmotCellElements, please refer to `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.

    Parameters
    ----------
    cellElementType
        The Marmot element which should be represented, e.g., CPE4.
    cellElementNumber
        The (unique) label of this CellElement.
    nodes
        The list of nodes for this CellElement.
    quadratureType
        The type of quadrature to be used.
    quadratureOrder
        The order of the quadrature to be used."""
    def __init__(self, cellType, cellNumber, nodes, quadratureType, quadratureOrder):
        super().__init__(cellType, cellNumber, nodes)

    @property
    def nMaterialPoints(self):
        return self._nMaterialPoints

    @property
    def elNumber(self):
        return self._cellNumber

    def getRequestedMaterialPointCoordinates(self, ):
        """Get the coordinates of the material points in the cell."""
        cdef int dim = self._nodeCoordinates.shape[1]
        cdef np.ndarray mpCoordinates = np.zeros( (self._nMaterialPoints, dim) )
        cdef double[:,::1] _coordsView = mpCoordinates
        self._marmotCellElement.getRequestedMaterialPointCoordinates(&_coordsView[0, 0])
        return mpCoordinates

    def getRequestedMaterialPointVolumes(self, ):
        """Get the volumes of the material points in the cell."""
        cdef np.ndarray vols = np.zeros(self._nMaterialPoints)
        cdef double[::1] _volsView = vols
        self._marmotCellElement.getRequestedMaterialPointVolumes(&_volsView[0])
        return vols

    def getRequestedMaterialPointType(self, ):
        """Get the type of the material points in the cell."""
        return MarmotMaterialPointWrapper

    def acceptLastState(self, ):
        pass
