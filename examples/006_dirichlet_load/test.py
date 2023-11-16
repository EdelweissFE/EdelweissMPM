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

import meshio

# from fe.steps.stepmanager import StepManager, StepActionDefinition, StepActionDefinition
from fe.journal.journal import Journal
from mpm.fields.nodefield import MPMNodeField
from mpm.fieldoutput.fieldoutput import MPMFieldOutputController

from mpm.generators import rectangulargridgenerator, rectangularmpgenerator
from mpm.mpmmanagers.simplempmmanager import SimpleMaterialPointManager

# from mpm.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from mpm.models.mpmmodel import MPMModel
from mpm.numerics.dofmanager import MPMDofManager
from mpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from mpm.sets.cellset import CellSet
from fe.sets.nodeset import NodeSet

# from fe.steps.adaptivestep import AdaptiveStep
from fe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from mpm.solvers.nqs import NonlinearQuasistaticSolver
from fe.linsolve.pardiso.pardiso import pardisoSolve
from mpm.stepactions.dirichlet import Dirichlet
from mpm.stepactions.bodyload import BodyLoad

import numpy as np


if __name__ == "__main__":
    dimension = 2

    journal = Journal()

    np.set_printoptions(precision=3)

    mpmModel = MPMModel(dimension)

    rectangulargridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=200.0,
        y0=0.0,
        h=100.0,
        nX=4,
        nY=2,
        cellProvider="marmot",
        cellType="Displacement/SmallStrain/Quad4",
    )
    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=10.0,
        l=180.0,
        y0=5.0,
        h=90.0,
        nX=10,
        nY=20,
        mpProvider="marmot",
        mpType="Displacement/SmallStrain/PlaneStrain",
    )

    material = "LINEARELASTIC"
    materialProperties = np.array([30000.0, 0.3])
    for mp in mpmModel.materialPoints.values():
        mp.assignMaterial(material, materialProperties)

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SimpleMaterialPointManager(allCells, allMPs)

    activeCells = None
    activeNodes = None

    journal.printSeperationLine()

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)

    nodeFieldOnAllCells = mpmModel.nodeFields["displacement"].subset(allCells)

    fieldOutputController.addPerNodeFieldOutput("dU", nodeFieldOnAllCells, "dU")
    fieldOutputController.addPerMaterialPointFieldOutput(
        "displacement",
        allMPs,
        "displacement",
    )
    fieldOutputController.addPerMaterialPointFieldOutput(
        "stress",
        allMPs,
        "stress",
    )
    fieldOutputController.addPerMaterialPointFieldOutput(
        "strain",
        allMPs,
        "strain",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", mpmModel, fieldOutputController, journal, None)

    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["dU"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["stress"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["strain"], create="perNode")

    ensightOutput.initializeJob()

    outputManagers = [
        ensightOutput,
    ]

    dirichletLeft = Dirichlet(
        "left",
        mpmModel.nodeSets["rectangular_grid_left"],
        "displacement",
        {0: 0.0, 1: 0.0},
        mpmModel,
        journal,
    )

    dirichletRight = Dirichlet(
        "right",
        mpmModel.nodeSets["rectangular_grid_right"],
        "displacement",
        {0: -20.0, 1: 0.0},
        mpmModel,
        journal,
    )

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-2, 1e-2, 1e-3, 1000, journal)

    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = dict()

    iterationOptions["nMaximumIterations"] = 5
    iterationOptions["nCrititcalIterations"] = 3
    iterationOptions["nAllowedResidualGrowths"] = 3

    linearSolver = pardisoSolve

    nonlinearSolver.solveStep(
        adaptiveTimeStepper,
        linearSolver,
        mpmManager,
        [dirichletLeft, dirichletRight],
        [],
        mpmModel,
        fieldOutputController,
        outputManagers,
        iterationOptions,
    )

    fieldOutputController.finalizeJob()
    ensightOutput.finalizeJob()
