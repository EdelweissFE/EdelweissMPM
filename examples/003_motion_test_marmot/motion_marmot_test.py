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

from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmpm.fields.nodefield import MPMNodeField
from edelweissmpm.generators import rectangulargridgenerator, rectangularmpgenerator
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.mpmmanagers.simplempmmanager import SimpleMaterialPointManager
from edelweissmpm.numerics.dofmanager import MPMDofManager
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager


def run_sim():
    dimension = 2

    journal = Journal()

    mpmModel = MPMModel(dimension)

    linearElasticMaterial = {"material": "LINEARELASTIC", "properties": np.array([30000.0, 0.3])}

    rectangulargridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=200.0,
        y0=0.0,
        h=100.0,
        nX=40,
        nY=20,
        cellProvider="LagrangianMarmotCell",
        cellType="Displacement/SmallStrain/Quad4",
    )
    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=5.0,
        l=20.0,
        y0=40.0,
        h=20.0,
        nX=3,
        nY=3,
        mpProvider="marmot",
        mpType="Displacement/SmallStrain/PlaneStrain",
        material=linearElasticMaterial,
    )

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

    try:
        for i in range(100):
            print("time step {:}".format(i))

            mpmManager.updateConnectivity()

            activeCells = mpmManager.getActiveCells()
            activeNodes = set([n for cell in activeCells for n in cell.nodes])

            print("active cells:")
            print([c.cellNumber for c in activeCells])

            print("active nodes:")
            print([n.label for n in activeNodes])

            for c in activeCells:
                print(
                    "cell {:} hosts material points {:}".format(
                        c.cellNumber, [mp.number for mp in c.assignedMaterialPoints]
                    )
                )

            activeNodeFields = {
                nodeField.name: MPMNodeField(nodeField.name, nodeField.dimension, activeNodes)
                for nodeField in mpmModel.nodeFields.values()
            }
            activeNodeFields["displacement"].createFieldValueEntry("dU")

            scalarVariables = []
            elements = []
            cellElements = []
            constraints = []
            nodeSets = []

            dofManager = MPMDofManager(
                activeNodeFields.values(), scalarVariables, elements, constraints, nodeSets, activeCells, cellElements
            )

            dofVector = dofManager.constructDofVector()

            time = 10 * i
            shift = np.asarray([2.0, 2.0 * np.cos(4 * np.pi * i / 100.0)])

            activeNodeFields["displacement"]["dU"][:] = shift
            dofManager.writeNodeFieldToDofVector(dofVector, activeNodeFields["displacement"], "dU")

            for c in activeCells:
                dUCell = dofVector[c]
                c.interpolateFieldsToMaterialPoints(dUCell)

            for mp in allMPs:
                mp.computeYourself(time, i)

            mpmModel.nodeFields["displacement"].copyEntriesFromOther(activeNodeFields["displacement"])

            mpmModel.advanceToTime(time)

            fieldOutputController.finalizeIncrement()

            ensightOutput.finalizeIncrement()

            journal.printSeperationLine()

    except Exception as e:
        print(str(e))
        raise

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()
        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = journal.linewidth
        print(prettytable)

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

    res = mpmModel.nodeFields["displacement"]["dU"]

    gold = np.loadtxt("gold.csv")

    assert np.isclose(res, gold).all()


if __name__ == "__main__":
    mpmModel = run_sim()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create-gold",
        dest="create_gold",
        action="store_true",
        help="create the gold file.",
    )
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", mpmModel.nodeFields["displacement"]["dU"])
