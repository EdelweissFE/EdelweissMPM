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

from edelweissfe.models.femodel import FEModel
from edelweissfe.journal.journal import Journal
from edelweissfe.variables.fieldvariable import FieldVariable
from edelweissmpm.sets.materialpointset import MaterialPointSet
from edelweissmpm.sets.cellset import CellSet
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissmpm.fields.nodefield import MPMNodeField
from edelweissfe.config.phenomena import phenomena, getFieldSize
from prettytable import PrettyTable


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
                        node.fields[field] = FieldVariable(node, field)

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

        super().prepareYourself(journal)

        # TODO move this to EdelweissFE
        for field in self.nodeFields.values():
            field.createFieldValueEntry("U")

    def advanceToTime(self, time: float):
        """Accept the current state of the model and sub instances, and
        set the new time.

        Parameters
        ----------
        time
            The new time.
        """

        for mp in self.materialPoints.values():
            mp.acceptStateAndPosition()

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

    def makePrettyTableSummary(self):
        prettytable = PrettyTable(("model property", ""))

        prettytable.add_row(("domain dim.", self.domainSize))
        prettytable.add_row(("time", self.time))
        prettytable.add_row(("nodes", len(self.nodes)))
        prettytable.add_row(("elements", len(self.elements)))
        prettytable.add_row(("node sets", list(self.nodeSets.keys())))
        prettytable.add_row(("node fields", list(self.nodeFields.keys())))
        prettytable.add_row(("element sets", list(self.elementSets.keys())))
        prettytable.add_row(("sections", list(self.sections.keys())))
        prettytable.add_row(("surfaces", list(self.surfaces.keys())))
        prettytable.add_row(("constraints", list(self.constraints.keys())))
        prettytable.add_row(("materials", list(self.materials.keys())))
        prettytable.add_row(("analytical fields", list(self.analyticalFields.keys())))
        prettytable.add_row(("scalar vars.", len(self.scalarVariables)))
        prettytable.add_row(("material points", len(self.materialPoints)))
        prettytable.add_row(("cells", len(self.cells)))
        prettytable.add_row(("material point sets", list(self.materialPointSets.keys())))
        prettytable.add_row(("cell sets", list(self.cellSets.keys())))

        prettytable.min_width["model property"] = 80
        prettytable.align = "l"

        return prettytable
