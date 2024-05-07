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
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from edelweissfe.utils.exceptions import CutbackRequest

from libc.stdlib cimport free, malloc
from libcpp.memory cimport allocator, make_unique, unique_ptr

from edelweissmpm.cellelements.marmotcellelement.marmotcellelement cimport (
    MarmotCell, MarmotCellElement, MarmotCellElementFactory,
    MarmotCellElementWrapper, MarmotMaterialPoint)
from edelweissmpm.materialpoints.marmotmaterialpoint.mp cimport \
    MarmotMaterialPointWrapper


@cython.final # no subclassing -> cpdef with nogil possible
cdef class LagrangianMarmotCellElementWrapper(MarmotCellElementWrapper):
    """This class is a wrapper for Lagrangian MarmotCellElements. It is used to create a MarmotCellElement from a list of nodes and to store the nodes of the cell element. It also provides a method to update the material points of the cell element."""

    def __init__(self, cellType, cellNumber, nodes, quadratureType, quadratureOrder):
        super().__init__(cellType, cellNumber, nodes, quadratureType, quadratureOrder)

    def __cinit__(self, str cellType, int cellNumber, list nodes, str quadratureType, int quadratureOrder):
        """This C-level method is responsible for actually creating the MarmotCellElement.

        Parameters
        ----------
        cellType : str
            The type of the cell element.
        cellNumber : int
            The number of the cell element.
        nodes : list
            The nodes of the cell element.
        quadratureType : str
            The type of the quadrature.
        quadratureOrder : int
            The order of the quadrature. """
        self._nodes = nodes

        cdef np.ndarray nodeCoordinates = np.array([ node.coordinates for node in nodes])
        self._nodeCoordinates = nodeCoordinates

        try:
            # create the MarmotCellElement
            self._marmotCell = self._marmotCellElement = MarmotCellElementFactory.createCellElement( cellType.encode('utf-8'), self._cellNumber, &self._nodeCoordinates[0,0], self._nodeCoordinates.size, quadratureType.encode('utf-8'), quadratureOrder)
        except IndexError:
            raise NotImplementedError("Marmot cellelement {:} not found in library.".format(cellType))

        self._nMaterialPoints = self._marmotCellElement.getNMaterialPoints()
    
    def __dealloc__(self):
        """This method is called when the object is deleted. It is responsible for freeing the memory of the MarmotCellElement."""
        if isinstance(self, LagrangianMarmotCellElementWrapper):
            del self._marmotCellElement
