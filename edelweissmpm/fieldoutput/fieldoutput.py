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
from typing import Callable

from edelweissfe.journal.journal import Journal
from edelweissfe.utils.fieldoutput import FieldOutputController, _FieldOutputBase

from edelweissmpm.fieldoutput.mpresultcollector import MaterialPointResultCollector
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.sets.materialpointset import MaterialPointSet


class MaterialPointFieldOutput(_FieldOutputBase):
    """
    A FieldOutput for material points.

    Parameters
    ----------
    name
        The name of this FieldOutput.
    mpSet
        The :class:`MaterialPointSet on which this FieldOutput operates.
    resultName
        The name of the result entry in the :class:`ElementBase.
    model
        The :class:`MPMModel tree instance.
    journal
        The :class:`Journal instance for logging.
    saveHistory
        If the history of the results should be saved.
    f_x
        The function to apply to the results.
    export
        Whether to export the results.
    fExport_x
        The function to apply to the exported results.
    """

    def __init__(
        self,
        name: str,
        mpSet: MaterialPointSet,
        resultName: str,
        model: MPMModel,
        journal: Journal,
        saveHistory: bool = False,
        f_x=Callable,
        export: str = None,
        fExport_x=Callable,
    ):
        self.associatedSet = mpSet
        self.resultName = resultName

        self.mpResultCollector = MaterialPointResultCollector(list(self.associatedSet), self.resultName)

        super().__init__(name, model, journal, saveHistory, f_x, export, fExport_x)

    def updateResults(self, model: MPMModel):
        """Update the field output.
        Will use the current solution and reaction vector if result is a nodal result.

        Parameters
        ----------
        model
            The model tree.
        """

        result = self.mpResultCollector.getCurrentResults()

        super()._applyResultsPipleline(result)


class MPMFieldOutputController(FieldOutputController):
    """
    The central module for managing field outputs, which can be used by output managers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def addPerMaterialPointFieldOutput(
        self,
        name: str,
        materialPointSet: MaterialPointSet,
        result: str = None,
        saveHistory: bool = False,
        f_x=None,
        export: str = None,
        fExport_x=None,
    ):
        """
        Parameters
        ----------
        name
            The name of this FieldOutput.
        nodeField
            The :class:`NodeField, on which this FieldOutput should operate.
        resultName
            The name of the result entry in the :class:`NodeField
        saveHistory
            If the history of the results should be saved.
        f_x
            The function to apply to the results.
        export
            Whether to export the results.
        fExport_x
            The function to apply to the exported results.
        """
        if name in self.fieldOutputs:
            raise Exception("FieldOutput {:} already exists!".format(name))

        if not result:
            result = name

        self.fieldOutputs[name] = MaterialPointFieldOutput(
            name, materialPointSet, result, self.model, self.journal, saveHistory, f_x, export, fExport_x
        )
