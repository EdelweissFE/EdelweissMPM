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
import pytest
import argparse

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
import fe.utils.performancetiming as performancetiming

import numpy as np


def run_sim():
    dimension = 2

    journal = Journal()

    np.set_printoptions(precision=3)

    mpmModel = MPMModel(dimension)

    rectangulargridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=200.0,
        y0=0.0,
        h=100.0,
        nX=20,
        nY=10,
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
    )

    material = "LINEARELASTIC"
    materialProperties = np.array([30000.0, 0.3])
    for mp in mpmModel.materialPoints.values():
        mp.assignMaterial(material, materialProperties)

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
            time = float(i)
            print("time step {:}".format(i))

            mpmManager.updateConnectivity()

            if mpmManager.hasChanged():
                print("material points in cells have changed since previous localization")

                activeCells = mpmManager.getActiveCells()
                activeNodes = set([n for cell in activeCells for n in cell.nodes])

                print("active cells:")
                print([c.cellNumber for c in activeCells])

                print("active nodes:")
                print([n.label for n in activeNodes])

                for c in activeCells:
                    print(
                        "cell {:} hosts material points {:}".format(
                            c.cellNumber, [mp.label for mp in c.assignedMaterialPoints]
                        )
                    )

                activeNodeFields = {
                    nodeField.name: MPMNodeField(nodeField.name, nodeField.dimension, activeNodes)
                    for nodeField in mpmModel.nodeFields.values()
                }
                activeNodeFields["displacement"].createFieldValueEntry("dU")

                scalarVariables = []
                elements = []
                constraints = []

                activeNodeSets = [
                    NodeSet(nodeSet.name, activeNodes.intersection(nodeSet)) for nodeSet in mpmModel.nodeSets.values()
                ]

                dofManager = MPMDofManager(
                    activeNodeFields.values(), scalarVariables, elements, constraints, activeNodeSets, activeCells
                )

                dUActiveCells = dofManager.constructDofVector()

            time = 10 * i
            shift = np.asarray([2.0, 6.0 * np.cos(4 * np.pi * i / 100.0)])
            activeNodeFields["displacement"]["dU"][:] = shift
            dofManager.writeNodeFieldToDofVector(dUActiveCells, activeNodeFields["displacement"], "dU")

            idcsTop = dofManager.idcsOfFieldsOnNodeSetsInDofVector["displacement"]["rectangular_grid_top"]
            dUActiveCells[idcsTop[1::2]] = 0.0

            idcsBottom = dofManager.idcsOfFieldsOnNodeSetsInDofVector["displacement"]["rectangular_grid_bottom"]
            dUActiveCells[idcsBottom[1::2]] = 0.0

            idcsRight = dofManager.idcsOfFieldsOnNodeSetsInDofVector["displacement"]["rectangular_grid_right"]
            dUActiveCells[idcsRight] = 0.0

            dofManager.writeDofVectorToNodeField(dUActiveCells, activeNodeFields["displacement"], "dU")

            for c in activeCells:
                dUCell = dUActiveCells[c]
                c.interpolateFieldsToMaterialPoints(dUCell)

            for mp in allMPs:
                mp.computeYourself(time, i)

            mpmModel.nodeFields["displacement"].copyEntriesFromOther(activeNodeFields["displacement"])

            mpmModel.advanceToTime(time)

            fieldOutputController.finalizeIncrement()

            print("material point stresses:")
            print(fieldOutputController.fieldOutputs["stress"].getLastResult())

            ensightOutput.finalizeIncrement()

            journal.printSeperationLine()

    except Exception as e:
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
    mpmModel = run_sim()

    res = mpmModel.nodeFields["displacement"]["dU"]

    gold = np.loadtxt("gold.csv")

    assert np.isclose(res, gold).all()


if __name__ == "__main__":
    mpmModel = run_sim()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", mpmModel.nodeFields["displacement"]["dU"])
