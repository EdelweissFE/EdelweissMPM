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

from fe.models.femodel import FEModel
from fe.journal.journal import Journal
from fe.variables.fieldvariable import FieldVariable
from mpm.sets.materialpointset import MaterialPointSet
from mpm.sets.cellset import CellSet
from fe.sets.elementset import ElementSet
from fe.sets.nodeset import NodeSet
from mpm.fields.nodefield import MPMNodeField
from fe.config.phenomena import phenomena, getFieldSize


class MPMModel(FEModel):
    """This is is a standard mpm model tree.
    It takes care of the correct number of variables,
    for nodes and scalar degrees of freedom, and it manages the fields.

    Parameters
    ----------
    dimension
        The dimension of the model.
    """

    identification = "MPMModel"

    def __init__(self, dimension: int):
        self.cells = {}  #: The collection of Cells in the present model.
        self.cellSets = {}  #: The collection of CellSets in the present model.
        self.materialPoints = {}  #: The collection of MaterialPoints in the present model.
        self.materialPointSets = {}  #: The collection of MaterialPointSets in the present model.

        super().__init__(dimension)

    def _populateNodeFieldVariablesFromCells(
        self,
    ):
        """Creates FieldVariables on GridNodes depending on the all
        MaterialPointCells .
        """
        for cell in self.cells.values():
            for node, nodeFields in zip(cell.nodes, cell.fields):
                for field in nodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node)

    def _prepareVariablesAndFields(self, journal):
        """Prepare all variables and fields for a simulation.

        Parameters
        ----------
        journal
            The journal instance.
        """

        journal.message("Activating fields on Nodes from Cells", self.identification)
        self._populateNodeFieldVariablesFromCells()

        return super()._prepareVariablesAndFields(journal)

    def _prepareMaterialPoints(self, journal: Journal):
        """Prepare elements for a simulation.
        In detail, sections are assigned.


        Parameters
        ----------
        journal
            The journal instance.
        """
        for section in self.sections.values():
            section.assignSectionPropertiesToModel(self)

        for mp in self.materialPoints.values():
            mp.initializeYourself()

    def prepareYourself(self, journal: Journal):
        """Prepare the model for a simulation.
        Creates the variables, bundles the fields,
        and initializes elements.


        Parameters
        ----------
        journal
            The journal instance.
        """

        self.nodeSets["all"] = NodeSet("all", self.nodes.values())
        self.elementSets["all"] = ElementSet("all", self.elements.values())
        self.materialPointSets["all"] = MaterialPointSet("all", self.materialPoints.values())
        self.cellSets["all"] = CellSet("all", self.cells.values())

        self._prepareMaterialPoints(journal)

        return super().prepareYourself(journal)

    def advanceToTime(self, time: float):
        """Accept the current state of the model and sub instances, and
        set the new time.

        Parameters
        ----------
        time
            The new time.
        """

        for mp in self.materialPoints.values():
            mp.acceptLastState()

        return super().advanceToTime(time)

    def _createNodeFieldsFromNodes(self, nodes: list, nodeSets: list) -> dict[str, MPMNodeField]:
        """Bundle nodal FieldVariables together in contiguous NodeFields.

        Parameters
        ----------
        nodes
            The list of Nodes from which the NodeFields should be created.
        nodeSets
            The list of NodeSets, which should be considered in the index map of the NodeFields.

        Returns
        -------
        dict[str,NodeField]
            The dictionary containing the NodeField instances for every active field."""

        domainSize = self.domainSize

        theNodeFields = dict()
        for field in phenomena.keys():
            fieldSize = getFieldSize(field, domainSize)

            theNodeField = MPMNodeField(field, getFieldSize(field, domainSize), nodes)

            if theNodeField.nodes:
                theNodeFields[field] = theNodeField

        return theNodeFields
