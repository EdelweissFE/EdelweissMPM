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
import argparse

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmpm.generators import boxbsplinegridgenerator, boxmpgenerator
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmpm.solvers.nqsmarmotparallel import NQSParallelForMarmot
from edelweissmpm.stepactions.bodyload import BodyLoad
from edelweissmpm.stepactions.dirichlet import Dirichlet


def run_sim(logFile=None, order: int = 2):
    dimension = 3

    journal = Journal()
    if logFile:
        journal.setFileOutput(logFile)

    np.set_printoptions(precision=3)

    mpmModel = MPMModel(dimension)

    boxbsplinegridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=60.0,
        y0=0.0,
        h=80.0001,
        z0=0.0,
        d=60.0,
        nX=1 * (4 - order),
        nY=2 * (4 - order),
        nZ=1 * (4 - order),
        cellProvider="BSplineMarmotCell",
        cellType="GradientEnhancedMicropolar/BSpline/3D/{:}".format(order),
        order=order,
    )

    gmNeoHooke = {
        "material": "GMDAMAGEDSHEARNEOHOOKE",
        "properties": np.array([300.0, 0.3, 1.0, 0.1, 0.2, 1.4999, 1.0]),
    }

    boxmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=1e-8,
        l=40,
        y0=1e-8,
        h=80.0,
        z0=1e-8,
        d=40.0,
        nX=8,
        nY=16,
        nZ=8,
        mpProvider="marmot",
        mpType="GradientEnhancedMicropolar/3D",
        material=gmNeoHooke,
    )

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")

    journal.printPrettyTable(mpmModel.makePrettyTableSummary(), "Model summary")

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
        {0: 0.0, 1: 0.0, 2: 0.0},
        mpmModel,
        journal,
    )

    dirichletBack = Dirichlet(
        "back",
        mpmModel.nodeSets["boxgrid_back"],
        "displacement",
        {2: 0.0},
        mpmModel,
        journal,
    )

    dirichletLeft = Dirichlet(
        "left",
        mpmModel.nodeSets["boxgrid_left"],
        "displacement",
        {
            0: 0.0,
        },
        mpmModel,
        journal,
    )

    gravityLoad = BodyLoad(
        "theGravity",
        mpmModel,
        journal,
        mpmModel.cells.values(),
        "BodyForce",
        np.array([0.0, 10 * -1.0, 0.0]),
    )

    nonlinearSolver = NQSParallelForMarmot(journal)

    iterationOptions = dict()

    linearSolver = pardisoSolve

    try:
        journal.printSeperationLine()
        journal.message("preconsolidation & gravity", "Step 1")
        nonlinearSolver.solveStep(
            AdaptiveTimeStepper(mpmModel.time, 1.0, 2e-1, 2e-1, 2e-1, 100, journal),
            linearSolver,
            mpmModel,
            fieldOutputController,
            mpmManagers=[mpmManager],
            dirichlets=[dirichletBottom, dirichletLeft, dirichletBack],
            bodyLoads=[gravityLoad],
            outputManagers=outputManagers,
            userIterationOptions=iterationOptions,
        )

        prettytable = performancetiming.makePrettyTable()
        journal.printPrettyTable(prettytable, "Summary Step 1")

    except StepFailed as e:
        journal.errorMessage(str(e), "StepFailed")
        raise

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

    return mpmModel


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim_order1():
    try:
        mpmModel = run_sim(None, 1)
    except ValueError as e:
        pytest.skip(str(e))
        return

    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    gold = np.loadtxt("gold_order_1.csv")

    print(res - gold)

    assert np.isclose(res, gold).all()


def test_sim_order2():
    try:
        mpmModel = run_sim(None, 2)
    except ValueError as e:
        pytest.skip(str(e))
        return

    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    gold = np.loadtxt("gold_order_2.csv")

    print(res - gold)

    assert np.isclose(res, gold).all()


def test_sim_order3():
    try:
        mpmModel = run_sim(None, 3)
    except ValueError as e:
        pytest.skip(str(e))
        return

    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    gold = np.loadtxt("gold_order_3.csv")

    print(res - gold)

    assert np.isclose(res, gold).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    parser.add_argument("order", type=int, help="order of the bsplines.")
    args = parser.parse_args()

    with open("bspline_marmot_cells_3d_log.txt", "w") as f:
        mpmModel = run_sim(f, args.order)

    if args.create_gold:
        gold = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
        np.savetxt("gold_order_{:}.csv".format(args.order), gold)
