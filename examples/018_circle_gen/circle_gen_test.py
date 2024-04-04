# hyperelasticity -*- coding: utf-8 -*-
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
import pytest
import argparse
import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissmpm.fields.nodefield import MPMNodeField
from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController

from edelweissmpm.generators import rectangulargridgenerator, circlempgenerator
from edelweissmpm.mpmmanagers.simplempmmanager import SimpleMaterialPointManager
from edelweissmpm.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.numerics.dofmanager import MPMDofManager
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmpm.sets.cellset import CellSet
from edelweissmpm.constraints.penaltyequalvalue import PenaltyEqualValue
from edelweissfe.sets.nodeset import NodeSet

from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissmpm.solvers.nqs import NonlinearQuasistaticSolver
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissmpm.stepactions.dirichlet import Dirichlet
from edelweissmpm.stepactions.distributedload import MaterialPointPointWiseDistributedLoad
from edelweissfe.utils.exceptions import StepFailed
import edelweissfe.utils.performancetiming as performancetiming

import numpy as np
from datetime import datetime


def run_sim():
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
        h=50.1,
        nX=12,
        nY=6,
        cellProvider="LagrangianMarmotCell",
        cellType="GradientEnhancedMicropolar/Quad4",
    )

    center = np.array([50.0, 0.0])

    circlempgenerator.generateModelData(
        mpmModel,
        journal,
        x0=center[0],
        y0=center[1],
        R=40.0,
        distance=2.0,
        angle=np.pi,
        mpProvider="marmot",
        mpType="GradientEnhancedMicropolar/PlaneStrain",
    )

    material = "GMDAMAGEDSHEARNEOHOOKE"
    materialProperties = np.array([300.0, 0.3, 1.0, 2, 4, 1.4999])
    for mp in mpmModel.materialPoints.values():
        mp.assignMaterial(material, materialProperties)

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SmartMaterialPointManager(allCells, allMPs, dimension, options={"KDTreeLevels": 10})

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
        "deformation gradient",
        allMPs,
        "deformation gradient",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", mpmModel, fieldOutputController, journal, None)

    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["dU"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perNode")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perNode"
    )

    ensightOutput.initializeJob()

    outputManagers = [
        ensightOutput,
    ]

    dirichletBottom = Dirichlet(
        "bottom",
        mpmModel.nodeSets["rectangular_grid_bottom"],
        "displacement",
        {0: 0, 1: 0.0},
        mpmModel,
        journal,
    )

    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm

    pressureLoads = [
        MaterialPointPointWiseDistributedLoad(
            "theSurfacePressure",
            mpmModel,
            journal,
            [mp],
            "Pressure",
            5e2 * normalize(center - mp.getCenterCoordinates()),
        )
        for mp in mpmModel.materialPointSets["planeCircle_sleeve"]
    ]

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-1, 1e-1, 1e-3, 1000, journal)

    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 3
    iterationOptions["allowed residual growths"] = 3

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            mpmManager,
            [
                dirichletBottom,
            ],
            [],
            pressureLoads,
            [],
            mpmModel,
            fieldOutputController,
            outputManagers,
            iterationOptions,
        )

    except StepFailed as e:
        raise
    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettyTable = performancetiming.makePrettyTable()
        journal.printPrettyTable(prettyTable, "circle_gen_test")

    np.savetxt("U.csv", fieldOutputController.fieldOutputs["displacement"].getLastResult())

    return mpmModel


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim():
    try:
        mpmModel = run_sim()
    except NotImplementedError as e:
        pytest.skip(str(e))
        return

    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    gold = np.loadtxt("gold.csv")

    print(res - gold)

    assert np.isclose(res, gold).all()


if __name__ == "__main__":
    mpmModel = run_sim()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        gold = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
        np.savetxt("gold.csv", gold)
