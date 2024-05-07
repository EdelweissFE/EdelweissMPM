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

from edelweissmpm.constraints.penaltyconstrainmp2node import PenaltyConstrainMP2Node
from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmpm.fields.nodefield import MPMNodeField
from edelweissmpm.generators import (
    rectangularbsplinegridgenerator,
    rectangularcellelementgridgenerator,
    rectangulargridgenerator,
    rectangularmpgenerator,
)
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmpm.numerics.dofmanager import MPMDofManager
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmpm.sets.cellset import CellSet
from edelweissmpm.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmpm.stepactions.bodyload import BodyLoad
from edelweissmpm.stepactions.dirichlet import Dirichlet


@performancetiming.timeit("simulation")
def run_sim():
    dimension = 2

    journal = Journal()

    np.set_printoptions(precision=3)

    mpmModel = MPMModel(dimension)

    from edelweissmpm.materialpoints.marmotmaterialpoint.mp import MarmotMaterialPointWrapper

    gmNeoHookean = {
        "material": "GMDAMAGEDSHEARNEOHOOKE",
        "properties": np.array([300.0, 0.2, 1.0, 1, 2, 1.4999]),
    }

    rectangularcellelementgridgenerator.generateModelData(
        mpmModel,
        journal,
        name="fem",
        x0=0.0,
        l=100.0,
        y0=0.0,
        h=25.0,
        nX=10,
        nY=10,
        cellelementProvider="LagrangianMarmotCellElement",
        cellelementType="GradientEnhancedMicropolar/Quad4",
        quadratureType="QGAUSS_LOBATTO",
        quadratureOrder=2,
        thickness=1.0,
        mpClass=MarmotMaterialPointWrapper,
        mpType="GradientEnhancedMicropolar/PlaneStrain",
        material=gmNeoHookean,
        firstCellElementNumber=1,
        firstNodeNumber=1,
    )

    bspline_order = 2
    rectangularbsplinegridgenerator.generateModelData(
        mpmModel,
        journal,
        name="mpm",
        x0=0.0,
        l=200.0,
        y0=-1.0,
        h=200.0,
        nX=14,
        nY=14,
        cellProvider="BSplineMarmotCell",
        cellType="GradientEnhancedMicropolar/BSpline/{:}".format(bspline_order),
        order=bspline_order,
        firstCellNumber=1,
        firstNodeNumber=1000,
    )

    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        name="mpmPoints",
        x0=101.00,
        l=98.0,
        y0=0.001,
        h=25,
        nX=48,
        nY=11,
        mpProvider="marmot",
        mpType="GradientEnhancedMicropolar/PlaneStrain",
        material=gmNeoHookean,
        firstMPNumber=1000,
    )

    mpmModel.prepareYourself(journal)

    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")
    mpmModel.nodeFields["displacement"].createFieldValueEntry("U")

    cellElements = mpmModel.cellElementSets["all"]
    mpmCells = mpmModel.cellSets["all"]
    cellMPs = mpmModel.materialPointSets["mpmPoints_all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SmartMaterialPointManager(mpmCells, cellMPs, dimension, options={"KDTreeLevels": 10})

    journal.printSeperationLine()

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)

    nodeFieldOnAllCellElements = mpmModel.nodeFields["displacement"].subset(cellElements)

    fieldOutputController.addPerNodeFieldOutput("dU", nodeFieldOnAllCellElements, "dU")
    fieldOutputController.addPerNodeFieldOutput("U", nodeFieldOnAllCellElements, "U")
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
        "ensight", mpmModel, fieldOutputController, journal, None, exportCellSetParts=False
    )

    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["dU"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["U"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perNode")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perNode"
    )

    ensightOutput.initializeJob()

    outputManagers = [
        ensightOutput,
    ]

    dirichletLeft = Dirichlet(
        "left_fem",
        mpmModel.nodeSets["fem_left"],
        "micro rotation",
        {0: -1 * np.pi},
        mpmModel,
        journal,
    )

    dirichletRightMPM = Dirichlet(
        "right_mpm",
        mpmModel.nodeSets["mpm_right"],
        "displacement",
        {0: 0.0, 1: 0.0},
        mpmModel,
        journal,
    )

    constraints = []

    vertCoords = np.array([mp.getCenterCoordinates()[1] for mp in mpmModel.materialPointSets["mpmPoints_left"]])
    sortedMPsLeft = [mp for _, mp in sorted(zip(vertCoords, mpmModel.materialPointSets["mpmPoints_left"]))]

    for masterNode, slaveMP in zip(mpmModel.nodeSets["fem_right"], sortedMPsLeft):
        constraints.append(
            PenaltyConstrainMP2Node(
                "PenaltyConstrainMP2Node", mpmModel, slaveMP, masterNode, "displacement", [0, 1], 1e6
            )
        )
        constraints.append(
            PenaltyConstrainMP2Node(
                "PenaltyConstrainMP2Node", mpmModel, slaveMP, masterNode, "micro rotation", [0], 1e6
            )
        )

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 5e-2, 5e-2, 1e-3, 1000, journal)

    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 5
    iterationOptions["allowed residual growths"] = 3
    iterationOptions["default absolute field correction tolerance"] = 1e-10

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            [mpmManager],
            [dirichletLeft, dirichletRightMPM],
            [],
            [],
            constraints,
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
