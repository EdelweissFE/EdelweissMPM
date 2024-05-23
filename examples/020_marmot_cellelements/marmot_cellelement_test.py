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

from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmpm.generators import rectangularcellelementgridgenerator
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmpm.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmpm.stepactions.dirichlet import Dirichlet


@performancetiming.timeit("simulation")
def run_sim():
    dimension = 2

    journal = Journal()

    np.set_printoptions(precision=3)

    mpmModel = MPMModel(dimension)

    from edelweissmpm.materialpoints.marmotmaterialpoint.mp import (
        MarmotMaterialPointWrapper,
    )

    gmNeoHookean = {
        "material": "GMDAMAGEDSHEARNEOHOOKE",
        "properties": np.array([30000.0, 0.3, 1.0, 2, 4, 1.4999]),
    }

    rectangularcellelementgridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=100.0,
        y0=0.0,
        h=100.0,
        nX=10,
        nY=10,
        cellelementProvider="LagrangianMarmotCellElement",
        nNodesPerCellElement=8,
        cellelementType="GradientEnhancedMicropolar/Quad8",
        quadratureType="QGAUSS",
        quadratureOrder=2,
        thickness=1.0,
        mpClass=MarmotMaterialPointWrapper,
        mpType="GradientEnhancedMicropolar/PlaneStrain",
        material=gmNeoHookean,
    )

    mpmModel.prepareYourself(journal)
    journal.printPrettyTable(mpmModel.makePrettyTableSummary(), "Model Summary")
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")
    mpmModel.nodeFields["displacement"].createFieldValueEntry("U")
    mpmModel.nodeFields["displacement"].createFieldValueEntry("P")

    allCellElements = mpmModel.cellElementSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    journal.printSeperationLine()

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)

    nodeFieldOnAllCellElements = mpmModel.nodeFields["displacement"].subset(allCellElements)

    fieldOutputController.addPerNodeFieldOutput("dU", nodeFieldOnAllCellElements)
    fieldOutputController.addPerNodeFieldOutput("U", nodeFieldOnAllCellElements)
    fieldOutputController.addPerMaterialPointFieldOutput(
        "displacement",
        allMPs,
    )

    fieldOutputController.addPerMaterialPointFieldOutput(
        "deformation gradient",
        allMPs,
    )

    fieldOutputController.addPerNodeFieldOutput(
        "P", mpmModel.nodeFields["displacement"].subset(mpmModel.nodeSets["rectangular_grid_left"])
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", mpmModel, fieldOutputController, journal, None)
    ensightOutput.createPerNodeOutput(fieldOutputController.fieldOutputs["dU"], varSize=3)
    ensightOutput.createPerNodeOutput(fieldOutputController.fieldOutputs["U"], varSize=3)
    ensightOutput.createPerNodeOutput(fieldOutputController.fieldOutputs["displacement"])
    ensightOutput.createPerNodeOutput(fieldOutputController.fieldOutputs["deformation gradient"])

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
        {0: 50.0, 1: -50.0},
        mpmModel,
        journal,
    )

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-0, 1e-0, 1e-0, 1000, journal)

    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = nonlinearSolver.validOptions.copy()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 10
    iterationOptions["allowed residual growths"] = 3

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            [],
            [
                dirichletLeft,
                dirichletRight,
            ],
            [],
            [],
            [],
            mpmModel,
            fieldOutputController,
            outputManagers,
            iterationOptions,
        )
    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettyTable = performancetiming.makePrettyTable()
        prettyTable.min_table_width = journal.linewidth
        journal.printPrettyTable(prettyTable, "PerfGraph")

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

    print("elapsed time: {:}".format(performancetiming.times["simulation"].time))

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        gold = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
        np.savetxt("gold.csv", gold)
