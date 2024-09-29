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

import fe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from fe.journal.journal import Journal
from fe.linsolve.pardiso.pardiso import pardisoSolve
from fe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from fe.utils.exceptions import StepFailed
from mpm.constraints.penaltyweakdirichlet import PenaltyWeakDirichlet
from mpm.fieldoutput.fieldoutput import MPMFieldOutputController
from mpm.generators import rectangulargridgenerator, rectangularmpgenerator
from mpm.models.mpmmodel import MPMModel
from mpm.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from mpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from mpm.solvers.nqsmarmotparallel import NQSParallelForMarmot
from mpm.stepactions.dirichlet import Dirichlet


@performancetiming.timeit("simulation")
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
        h=100.0,
        nX=4 * 48,
        nY=4 * 48,
        cellProvider="LagrangianMarmotCell",
        cellType="GradientEnhancedMicropolar/Quad4",
    )
    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.01,
        l=79.98,
        y0=60.0,
        h=20.0,
        nX=4 * 240,
        nY=4 * 60,
        mpProvider="marmot",
        mpType="GradientEnhancedMicropolar/PlaneStrain",
    )

    material = "gmdamagedshearneohooke"
    materialProperties = np.array([30000.0, 0.3, 1.0, 2, 4, 1.4999])
    for mp in mpmModel.materialPoints.values():
        mp.assignMaterial(material, materialProperties)

    mpmModel.prepareYourself(journal)
    mpmModel.gridnodeFields["displacement"].createFieldValueEntry("dU")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SmartMaterialPointManager(allCells, allMPs, options={"KDTreeLevels": 10})

    journal.printSeperationLine()

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)

    gridnodeFieldOnAllCells = mpmModel.gridnodeFields["displacement"].subset(allCells)

    fieldOutputController.addPerGridNodeFieldOutput("dU", gridnodeFieldOnAllCells, "dU")
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

    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["dU"], create="perGridNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perGridNode")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perGridNode"
    )

    ensightOutput.initializeJob()

    outputManagers = [
        ensightOutput,
    ]

    dirichletLeft = Dirichlet(
        "left",
        mpmModel.gridnodeSets["rectangular_grid_left"],
        "displacement",
        {0: 0.0, 1: 0.0},
        mpmModel,
        journal,
    )

    weakDirichlets = [
        PenaltyWeakDirichlet(
            "weak dirichlet {:}".format(mp.label),
            mpmModel,
            mp,
            "displacement",
            {
                0: -0,
                1: -50.0,
            },
            1e8,
        )
        for mp in mpmModel.materialPointSets["rectangular_grid_right"]
    ]

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-2, 1e-2, 1e-3, 100, journal)

    nonlinearSolver = NQSParallelForMarmot(journal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 5
    iterationOptions["allowed residual growths"] = 3

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            mpmManager,
            [
                dirichletLeft,
            ],
            [],
            weakDirichlets,
            mpmModel,
            fieldOutputController,
            outputManagers,
            iterationOptions,
        )

    except StepFailed as e:
        print(e)
    finally:
        table = []
        k1 = "solve step"
        v1 = performancetiming.times[k1]
        table.append(("{:}".format(k1), "{:10.4f}s".format(v1.time)))
        for k2, v2 in v1.items():
            table.append(("  {:}".format(k2), "  {:10.4f}s".format(v2.time)))
            for k3, v3 in v2.items():
                table.append(("    {:}".format(k3), "    {:10.4f}s".format(v3.time)))

        journal.printTable(
            table,
            "step 1",
        )

    fieldOutputController.finalizeJob()
    ensightOutput.finalizeJob()

    # np.savetxt("U.csv", fieldOutputController.fieldOutputs["displacement"].getLastResult())

    return mpmModel


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim():
    mpmModel = run_sim()

    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    gold = np.loadtxt("gold.csv")

    print(res - gold)

    assert np.isclose(res, gold).all()


if __name__ == "__main__":
    mpmModel = run_sim()

    print("elapsed time: {:}".format(performancetiming.times["simulation"].time))

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        gold = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
        np.savetxt("gold.csv", gold)
