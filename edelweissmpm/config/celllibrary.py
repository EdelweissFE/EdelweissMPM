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
EdelweissFE currently supports finite element implementations provided by the Marmot library.
In future, elements by other providers or elements directly implemented in EdelweissFE may be added here.

.. code-block:: edelweiss
    :caption: Example:

    *element, type=C3D8, provider=marmot
        ** el_label, node1, node2, node3, node4, ... 
        1000,        1,     2,     3,     4,     ...
"""


def getCellClass(provider: str) -> type:
    """Get the class type of the requested element provider.

    Parameters
    ----------
    provider
        The name of the cell provider to load.

    Returns
    -------
    type
        The cell provider class type.
    """

    if provider.lower() == "test":
        from edelweissmpm.cells.test.cell import Cell

        return Cell

    if provider.lower() == "LagrangianMarmotCell".lower():
        from edelweissmpm.cells.marmotcell.lagrangianmarmotcell import LagrangianMarmotCellWrapper

        return LagrangianMarmotCellWrapper

    if provider.lower() == "BSplineMarmotCell".lower():
        from edelweissmpm.cells.marmotcell.bsplinemarmotcell import BSplineMarmotCellWrapper

        return BSplineMarmotCellWrapper
