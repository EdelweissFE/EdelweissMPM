#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____ 
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|  
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___ 
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#                                                       
# 
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
# 
#  Matthias Neuner matthias.neuner@uibk.ac.at
# 
#  This file is part of EdelweissFE.
# 
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
# 
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
cimport numpy as np
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np

from edelweissmpm.cells.marmotcell.marmotcell cimport (MarmotCell,
                                                       MarmotCellWrapper)


cdef extern from "Marmot/MarmotMPMLibrary.h" namespace "MarmotLibrary" nogil:
    
    cdef cppclass MarmotCellElementFactory:
        @staticmethod
        MarmotCellElement* createCellElement(const string& cellElementName, 
                               int cellElementNumber,
                               const double* nodeCoordinates,
                               int sizeGridNodeCoordinates,
                                const string& quadratureType,
                                int quadratureOrder) except +ValueError

cdef extern from "Marmot/MarmotMaterialPoint.h" nogil:
    cdef cppclass MarmotMaterialPoint:
        pass

cdef extern from "Marmot/MarmotCellElement.h":
    cdef cppclass MarmotCellElement(MarmotCell) nogil:
        pass
        
        int getNMaterialPoints() 

        void getRequestedMaterialPointCoordinates( double* coordinates ) 

        void getRequestedMaterialPointVolumes( double* volumes ) 

        
cdef class MarmotCellElementWrapper(MarmotCellWrapper):
    
    cdef int _nMaterialPoints
    cdef MarmotCellElement* _marmotCellElement
