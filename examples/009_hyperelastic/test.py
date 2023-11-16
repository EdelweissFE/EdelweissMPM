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
import numpy as np

from fe.journal.journal import Journal
from mpm.fields.nodefield import MPMNodeField
from mpm.fieldoutput.fieldoutput import MPMFieldOutputController

from mpm.generators import rectangulargridgenerator, rectangularmpgenerator
from mpm.mpmmanagers.simplempmmanager import SimpleMaterialPointManager
from mpm.models.mpmmodel import MPMModel
from mpm.numerics.dofmanager import MPMDofManager
from mpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from mpm.sets.cellset import CellSet
from fe.sets.nodeset import NodeSet

from fe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from mpm.solvers.nqs import NonlinearQuasistaticSolver
from fe.linsolve.pardiso.pardiso import pardisoSolve
from mpm.stepactions.dirichlet import Dirichlet
from mpm.stepactions.bodyload import BodyLoad
from fe.utils.exceptions import StepFailed

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
        l=100.0,
        y0=0.0,
        h=100.0,
        nX=12,
        nY=12,
        cellProvider="marmot",
        cellType="GradientEnhancedMicropolar/Quad4",
    )
    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.01,
        l=99.98,
        y0=60.0,
        h=20.0,
        nX=60,
        nY=15,
        mpProvider="marmot",
        mpType="GradientEnhancedMicropolar/PlaneStrain",
    )

    material = "GMDAMAGEDSHEARNEOHOOKE"
    # material = "LINEARELASTIC"
    materialProperties = np.array([30000.0, 0.3, 0, 1e-9, 2e-9, 1.4999])
    # materialProperties = np.array([30000.0, 0.3, ])
    for mp in mpmModel.materialPoints.values():
        mp.assignMaterial(material, materialProperties)

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SimpleMaterialPointManager(allCells, allMPs)

    journal.printSeperationLine()

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)

    nodeFieldOnAllCells = mpmModel.nodeFields["displacement"].subset(allCells)

    fieldOutputController.addPerNodeFieldOutput("dU", nodeFieldOnAllCells, "dU")
    fieldOutputController.addPerMaterialPointFieldOutput(
        "displacement",
        allMPs,
        "displacement",
    )
    # fieldOutputController.addPerMaterialPointFieldOutput(
    #     "stress",
    #     allMPs,
    #     "stress",
    # )
    # fieldOutputController.addPerMaterialPointFieldOutput(
    #     "deformation gradient",
    #     allMPs,
    #     "deformation gradient",
    # )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", mpmModel, fieldOutputController, journal, None)

    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["dU"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perNode")
    # ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perNode")
    # ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["stress"], create="perNode")
    # ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["strain"], create="perNode")

    ensightOutput.initializeJob()

    outputManagers = [
        ensightOutput,
    ]

    dirichletBottom = Dirichlet(
        "bottom",
        mpmModel.nodeSets["rectangular_grid_bottom"],
        "displacement",
        {1: 0.0},
        mpmModel,
        journal,
    )

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
        {0: 0.0, 1: 0.0},
        mpmModel,
        journal,
    )

    gravityLoad = BodyLoad(
        "theGravity",
        mpmModel,
        journal,
        mpmModel.cells.values(),
        "BodyForce",
        np.array([0.0, -200.0]),
    )

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-2, 1e-2, 1e-2, 1000, journal)

    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = dict()

    iterationOptions["nMaximumIterations"] = 5
    iterationOptions["nCrititcalIterations"] = 3
    iterationOptions["nAllowedResidualGrowths"] = 3

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            mpmManager,
            [dirichletBottom, dirichletLeft, dirichletRight],
            [gravityLoad],
            mpmModel,
            fieldOutputController,
            outputManagers,
            iterationOptions,
        )

    except StepFailed as e:
        print(e)

    fieldOutputController.finalizeJob()
    ensightOutput.finalizeJob()

    np.savetxt("U.csv", fieldOutputController.fieldOutputs["displacement"].getLastResult())
