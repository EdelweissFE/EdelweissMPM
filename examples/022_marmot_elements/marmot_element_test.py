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

    gmNeoHookean = {
        "material": "GMDAMAGEDSHEARNEOHOOKE",
        "name": "GMDAMAGEDSHEARNEOHOOKE",
        "properties": np.array([30000.0, 0.3, 1.0, 2, 4, 1.4999]),
    }

    mpmModel.materials["gmp"] = gmNeoHookean

    from edelweissfe.generators.planerectquad import generateModelData

    mpmModel = generateModelData(
        {
            "data": [
                "x0=0.0",
                "l=100.0",
                "y0=0.0",
                "h=100.0",
                "nX= 10",
                "nY= 10",
                "elProvider=marmot",
                "elType=GMCPE8RUL",
            ]
        },
        mpmModel,
        journal,
    )

    from edelweissfe.sections.plane import Section

    planeSec = Section("theSection", ["planeRect_all"], "gmp", mpmModel, thickness=1.0)

    mpmModel.sections["theSection"] = planeSec

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")
    mpmModel.nodeFields["displacement"].createFieldValueEntry("U")
    mpmModel.nodeFields["displacement"].createFieldValueEntry("P")

    journal.printSeperationLine()

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)
    fieldOutputController.addPerNodeFieldOutput(
        "U", mpmModel.nodeFields["displacement"].subset(mpmModel.elementSets["planeRect_all"]), "U"
    )
    fieldOutputController.addPerNodeFieldOutput(
        "P", mpmModel.nodeFields["displacement"].subset(mpmModel.nodeSets["planeRect_left"]), "P"
    )
    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", mpmModel, fieldOutputController, journal, None)
    ensightOutput.createPerNodeOutput(fieldOutputController.fieldOutputs["U"], varSize=3)
    ensightOutput.initializeJob()

    outputManagers = [
        ensightOutput,
    ]

    dirichletLeft = Dirichlet(
        "left_fem",
        mpmModel.nodeSets["planeRect_left"],
        "displacement",
        {0: 0, 1: 0},
        mpmModel,
        journal,
    )

    dirichletRight = Dirichlet(
        "right",
        mpmModel.nodeSets["planeRect_right"],
        "displacement",
        {0: 50.0, 1: -50.0},
        mpmModel,
        journal,
    )

    for el in mpmModel.elements.values():
        emptDef = np.array([0.0])
        el.setInitialCondition("initialize material", emptDef)

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-0, 1e-0, 1e-0, 1000, journal)

    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 10
    iterationOptions["allowed residual growths"] = 3

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            mpmModel,
            fieldOutputController,
            dirichlets=[
                dirichletLeft,
                dirichletRight,
            ],
            outputManagers=outputManagers,
            userIterationOptions=iterationOptions,
        )
    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettyTable = performancetiming.makePrettyTable()
        prettyTable.min_table_width = journal.linewidth
        journal.printPrettyTable(prettyTable, "PerfGraph")

    np.savetxt("U.csv", fieldOutputController.fieldOutputs["U"].getLastResult())

    return mpmModel, fieldOutputController


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim():
    try:
        mpmModel, fieldOutputController = run_sim()
    except ValueError as e:
        pytest.skip(str(e))
        return

    res = fieldOutputController.fieldOutputs["U"].getLastResult()
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
