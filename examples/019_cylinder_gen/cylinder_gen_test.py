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
import argparse
from datetime import datetime

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmpm.constraints.penaltyequalvalue import PenaltyEqualValue
from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmpm.fields.nodefield import MPMNodeField
from edelweissmpm.generators import boxbsplinegridgenerator, cylindermpgenerator
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.mpmmanagers.simplempmmanager import SimpleMaterialPointManager
from edelweissmpm.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmpm.numerics.dofmanager import MPMDofManager
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmpm.sets.cellset import CellSet
from edelweissmpm.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmpm.stepactions.dirichlet import Dirichlet
from edelweissmpm.stepactions.distributedload import MaterialPointPointWiseDistributedLoad


def run_sim():
    dimension = 3

    journal = Journal()

    np.set_printoptions(precision=3)

    mpmModel = MPMModel(dimension)

    boxbsplinegridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=100.01,
        y0=0.0,
        h=100.01,
        z0=0.0,
        d=100.01,
        nX=10,
        nY=10,
        nZ=10,
        cellProvider="BSplineMarmotCell",
        cellType="GradientEnhancedMicropolar/BSpline/3D/1",
        order=1,
    )

    center = np.array([50.0 + 1e-3, 1e-3, 1e-3])
    cylinderHeight = 100.0
    cylinderRadius = 50.0
    gmNeoHookean = {"material": "GMDAMAGEDSHEARNEOHOOKE", "properties": np.array([300.0, 0.3, 1.0, 2, 4, 1.4999])}

    cylindermpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=center[0],
        y0=center[1],
        z0=center[2],
        R=cylinderRadius,
        distance=5,
        angle=np.pi,
        H=cylinderHeight,
        mpProvider="marmot",
        mpType="GradientEnhancedMicropolar/3D",
        material=gmNeoHookean,
    )

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")
    mpmModel.nodeFields["displacement"].createFieldValueEntry("P")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SmartMaterialPointManager(allCells, allMPs, dimension, options={"KDTreeLevels": 10})

    journal.printSeperationLine()

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)

    bottomNodeField = mpmModel.nodeFields["displacement"].subset(mpmModel.nodeSets["boxgrid_bottom"])

    fieldOutputController.addPerNodeFieldOutput("reaction_force", bottomNodeField, "P", **{"f(x)": "np.sum(x, axis=0)"})
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

    ensightOutput = EnsightOutputManager(
        "ensight",
        mpmModel,
        fieldOutputController,
        journal,
        None,
        exportCellSetParts=False,
    )

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
        mpmModel.nodeSets["boxgrid_bottom"],
        "displacement",
        {1: 0},
        mpmModel,
        journal,
    )

    dirichletLeftBottom = Dirichlet(
        "leftBottom",
        mpmModel.nodeSets["boxgrid_leftBottom"],
        "displacement",
        {0: 0.0},
        mpmModel,
        journal,
    )

    dirichletBack = Dirichlet(
        "back",
        mpmModel.nodeSets["boxgrid_back"],
        "displacement",
        {2: 0},
        mpmModel,
        journal,
    )

    dirichletFront = Dirichlet(
        "front",
        mpmModel.nodeSets["boxgrid_front"],
        "displacement",
        {2: 0},
        mpmModel,
        journal,
    )

    # are for each mp for the half cylinder
    sleeveArea = cylinderRadius * np.pi * cylinderHeight
    areaPerMP = sleeveArea / len(mpmModel.materialPointSets["cylinder_sleeve"])
    pressure = 1e2

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
            pressure * areaPerMP * normalize((center - mp.getCenterCoordinates()) * [1, 1, 0]),
        )
        for mp in mpmModel.materialPointSets["cylinder_sleeve"]
    ]

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-0, 1e-0, 1e-3, 1000, journal)

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
            [mpmManager],
            [dirichletBottom, dirichletLeftBottom, dirichletBack, dirichletFront],
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
        journal.printPrettyTable(prettyTable, "cylinder_gen_test")

    journal.message(
        "Numerically integrated pressure y dir.: {:}".format(
            fieldOutputController.fieldOutputs["reaction_force"].getLastResult()[1]
        ),
        "plausibility check",
    )
    journal.message(
        "Analytically computed force y dir.: {:}".format(pressure * cylinderHeight * 2 * cylinderRadius),
        "plausibility check",
    )

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
    except ValueError as e:
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
