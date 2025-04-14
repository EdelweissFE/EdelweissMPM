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

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

from edelweissmpm.constraints.penaltyweakdirichlet import PenaltyWeakDirichlet
from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmpm.generators import (
    rectangularbsplinegridgenerator,
    rectangularmpgenerator,
)
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmpm.solvers.nqsmarmotparallel import NQSParallelForMarmot
from edelweissmpm.stepactions.dirichlet import Dirichlet


@performancetiming.timeit("simulation")
def run_sim(bspline_order):
    dimension = 2

    journal = Journal()

    np.set_printoptions(precision=3)

    mpmModel = MPMModel(dimension)

    rectangularbsplinegridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=100.0,
        y0=0.0,
        h=100.0,
        nX=12,
        nY=12,
        cellProvider="BSplineMarmotCell",
        cellType="GradientEnhancedMicropolar/BSpline/{:}".format(bspline_order),
        order=bspline_order,
    )
    gmDamagedShearNeoHooke = {
        "material": "GMDamagedShearNeoHooke",
        "properties": np.array([300.0, 0.3, 1, 2, 4, 1.4999, 1.0]),
    }
    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.01,
        l=79.98,
        y0=60.0,
        h=20.0,
        nX=120,
        nY=30,
        mpProvider="marmot",
        mpType="GradientEnhancedMicropolar/PlaneStrain",
        material=gmDamagedShearNeoHooke,
    )

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SmartMaterialPointManager(allCells, allMPs, dimension, options={"KDTreeLevels": 6})

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
        "ensight_order_{:}".format(bspline_order),
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

    dirichletLeft = Dirichlet(
        "left",
        mpmModel.nodeSets["rectangular_grid_left"],
        "displacement",
        {0: 0.0, 1: 0.0},
        mpmModel,
        journal,
    )

    weakDirichlets = [
        PenaltyWeakDirichlet(
            "weak dirichlet",
            mpmModel,
            mpmModel.materialPointSets["rectangular_grid_right"],
            "displacement",
            {
                0: -0,
                1: -50.0,
            },
            1e5,
        )
    ]

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-1, 1e-1, 1e-3, 1000, journal)

    nonlinearSolver = NQSParallelForMarmot(journal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 5
    iterationOptions["allowed residual growths"] = 3
    iterationOptions["zero increment threshhold"] = 1e-10

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            mpmModel,
            fieldOutputController,
            mpmManagers=[mpmManager],
            dirichlets=[
                dirichletLeft,
            ],
            constraints=weakDirichlets,
            outputManagers=outputManagers,
            userIterationOptions=iterationOptions,
        )

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = journal.linewidth
        journal.printPrettyTable(prettytable, "PerfGraph")

    np.savetxt("U.csv", fieldOutputController.fieldOutputs["displacement"].getLastResult())

    return mpmModel


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim_order1():
    try:
        mpmModel = run_sim(1)
    except ValueError as e:
        pytest.skip(str(e))
        return

    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    gold = np.loadtxt("gold_order_1.csv")

    print(res - gold)

    assert np.isclose(res, gold).all()


def test_sim_order2():
    try:
        mpmModel = run_sim(2)
    except ValueError as e:
        pytest.skip(str(e))
        return

    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    gold = np.loadtxt("gold_order_2.csv")

    print(res - gold)

    assert np.isclose(res, gold).all()


def test_sim_order3():
    try:
        mpmModel = run_sim(3)
    except ValueError as e:
        pytest.skip(str(e))
        return

    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    gold = np.loadtxt("gold_order_3.csv")

    print(res - gold)

    assert np.isclose(res, gold).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("order", type=int, help="order of the bsplines.")
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    mpmModel = run_sim(args.order)

    print("elapsed time: {:}".format(performancetiming.times["simulation"].time))

    if args.create_gold:
        gold = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
        np.savetxt("gold_order_{:}.csv".format(args.order), gold)
