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

import copy

from edelweissfe.config.phenomena import getFieldSize, phenomena
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.variables.fieldvariable import FieldVariable
from prettytable import PrettyTable

from edelweissmpm.fields.nodefield import MPMNodeField
from edelweissmpm.sets.cellelementset import CellElementSet
from edelweissmpm.sets.cellset import CellSet
from edelweissmpm.sets.materialpointset import MaterialPointSet
from edelweissmpm.sets.particleset import ParticleSet


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
        self.cellElements = {}  #: The collection of CellElements in the present model.
        self.cellElementSets = {}  #: The collection of CellElementSets in the present model.
        self.particles = {}  #: The collection of Particles in the present model.
        self.particleSets = {}  #: The collection of ParticleSets in the present model.
        self.meshfreeKernelFunctions = {}  #: The collection of MeshfreeKernelFunctions in the present model.
        self.particleKernelDomains = {}  #: The collection of ParticleKernelDomains in the present model.

        super().__init__(dimension)

    def _populateNodeFieldVariablesFromCells(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all
        MaterialPointCells .
        """
        for cell in self.cells.values():
            for node, nodeFields in zip(cell.nodes, cell.fields):
                for field in nodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node, field)

    def _populateNodeFieldVariablesFromCellElements(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all
        MaterialPointCells .
        """
        for cellElement in self.cellElements.values():
            for node, nodeFields in zip(cellElement.nodes, cellElement.fields):
                for field in nodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node, field)

    def _populateNodeFieldVariablesFromParticleKernelDomains(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all defined particle-kernel interactions.
        For particles and kernels, the situation is a bit more complicated compared to elements or cells:
        Unlike for elements or cells, there is no direct connection between a particle and a node.
        Hence, the fields on the nodes are not necessarily known at the beginning of simulation.

        For this reason, we force the user to tell us which particles (which have fields) are interacting with which kernelfunctions during the simulation.
        """

        for particleKernelInteraction in self.particleKernelDomains.values():
            theFields = particleKernelInteraction.particles[
                0
            ].baseFields  # TODO: We assume that all particles have the same fields; let's change this later
            for field in theFields:
                for kf in particleKernelInteraction.meshfreeKernelFunctions:
                    if field not in kf.node.fields:
                        kf.node.fields[field] = FieldVariable(kf.node, field)

    def _prepareVariablesAndFields(self, journal):
        """Prepare all variables and fields for a simulation.

        Parameters
        ----------
        journal
            The journal instance.
        """
        if self.cells:
            journal.message("Activating fields on Nodes from Cells", self.identification)
            self._populateNodeFieldVariablesFromCells()
        if self.cellElements:
            journal.message("Activating fields on Nodes from CellElements", self.identification)
            self._populateNodeFieldVariablesFromCellElements()
        if self.particleKernelDomains:
            journal.message("Activating fields on Nodes from Particles", self.identification)
            self._populateNodeFieldVariablesFromParticleKernelDomains()

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

    def _prepareParticles(self, journal: Journal):
        """Prepare elements for a simulation.
        In detail, sections are assigned.


        Parameters
        ----------
        journal
            The journal instance.
        """
        # for section in self.sections.values():
        #     section.assignSectionPropertiesToModel(self)

        for p in self.particles.values():
            p.initializeYourself()

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
        self.cellElementSets["all"] = CellElementSet("all", self.cellElements.values())
        self.particleSets["all"] = ParticleSet("all", self.particles.values())

        self._prepareMaterialPoints(journal)
        self._prepareParticles(journal)

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

        for p in self.particles.values():
            p.acceptStateAndPosition()

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
            theNodeField = MPMNodeField(field, getFieldSize(field, domainSize), nodes)

            if theNodeField.nodes:
                theNodeFields[field] = theNodeField

        return theNodeFields

    def getActiveSubModel(self):
        """
        For many applications only a part of the model is active.
        For instance, in MPM simulations only cells occupied by material points are active.
        In order to create a numerical model, we need to know which part of the model is active, and which is not.
        This function returns the active sub model, based on the current state of the model.

        Returns
        -------
        MPMModel
            The active sub model.
        """

        activeModel = copy.copy(self)

        activeCells = {cell.number: cell for mp in self.materialPoints.values() for cell in mp.cells}

        activeNodesWithPersistentFieldValues = set(
            n for element in self.elements.values() for n in element.nodes
        ) | set(n for element in self.cellElements.values() for n in element.nodes)

        activeNodesWithVolatileFieldValues = set(n for cell in activeCells for n in cell.nodes)

        activeNodesWithVolatileFieldValues |= set(
            kf.node for particle in self.particles.values() for kf in particle.kernelFunctions
        )

        activeNodes = activeNodesWithVolatileFieldValues | activeNodesWithPersistentFieldValues

        activeNodes = NodeSet("activeNodes", activeNodes)
        activeNodesWithPersistentFieldValues = NodeSet(
            "activeNodesWithPersistentFieldvalues", activeNodesWithPersistentFieldValues
        )
        activeNodesWithVolatileFieldValues = NodeSet(
            "activeNodesWithVolatileFieldValues", activeNodesWithVolatileFieldValues
        )

        reducedNodeFields = {
            nodeField.name: MPMNodeField(nodeField.name, nodeField.dimension, activeNodes)
            for nodeField in self.nodeFields.values()
        }

        reducedNodeSets = {
            nodeSet: NodeSet(nodeSet.name, set(activeNodes).intersection(nodeSet)) for nodeSet in self.nodeSets.values()
        }

        activeModel.nodeSets = reducedNodeSets
        activeModel.nodeFields = reducedNodeFields
        activeModel.cells = activeCells

        return activeModel

    def makePrettyTableSummary(self):
        """Create a pretty table with a summary of the model properties."""
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
        prettytable.add_row(("cell elements", len(self.cellElements)))
        prettytable.add_row(("material point sets", list(self.materialPointSets.keys())))
        prettytable.add_row(("cell sets", list(self.cellSets.keys())))
        prettytable.add_row(("cell element sets", list(self.cellElementSets.keys())))
        prettytable.add_row(("particles", len(self.particles)))
        prettytable.add_row(("particle sets", list(self.particleSets.keys())))
        prettytable.add_row(("meshfree kernel functions", len(self.meshfreeKernelFunctions.keys())))
        prettytable.add_row(("particle kernel domains", list(self.particleKernelDomains.keys())))

        prettytable.min_width["model property"] = 80
        prettytable.align = "l"

        return prettytable
