import meshio

from fe.steps.stepmanager import StepManager, StepActionDefinition, StepActionDefinition
from fe.journal.journal import Journal
from mpm.fields.nodefield import MPMNodeField
from mpm.fieldoutput.fieldoutput import MPMFieldOutputController

from mpm.generators import rectangulargridgenerator, rectangularmpgenerator
from mpm.mpmmanagers.simplempmmanager import SimpleMaterialPointManager
from mpm.mpmmanagers.smartmpmmanager import SmartMaterialPointManager, KDTree
from mpm.models.mpmmodel import MPMModel
from mpm.numerics.dofmanager import MPMDofManager
from mpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from mpm.sets.cellset import CellSet

import numpy as np


if __name__ == "__main__":
    dimension = 2

    journal = Journal()

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
        cellProvider="test",
        cellType="a dummy cell",
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
        mpProvider="test",
        mpType="a dummy mp",
    )

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SmartMaterialPointManager(allCells, allMPs, options= {"KDTreeLevels": 3})

    activeCells = None
    activeNodes = None

    journal.printSeperationLine()

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)

    nodeFieldOnAllCells = mpmModel.nodeFields["displacement"].subset(allCells)

    fieldOutputController.addPerNodeFieldOutput("dU", nodeFieldOnAllCells, "dU")
    fieldOutputController.addPerMaterialPointFieldOutput(
        "displacement", allMPs, "displacement", **{"f(x)": "np.pad(x,((0,0),(0,1)))"}
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", mpmModel, fieldOutputController, journal, None)

    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["dU"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perNode")

    for i in range(100):
        print("time step {:}".format(i))

        mpmManager.updateConnectivity()

        if mpmManager.hasChanged():
            print("material points in cells have changed since previous localization")

            if mpmManager.hasLostMaterialPoints():
                print("we have lost material points outside the grid!")
                break

            activeCells = mpmManager.getActiveCells()
            activeNodes = set([n for cell in activeCells for n in cell.nodes])

            print("active cells:")
            print([c.cellNumber for c in activeCells])

            print("active nodes:")
            print([n.label for n in activeNodes])

            for c in activeCells:
                print(
                    "cell {:} hosts material points {:}".format(
                        c.cellNumber, [mp.label for mp in mpmManager.getMaterialPointsInCell(c)]
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
            nodeSets = []

            dofManager = MPMDofManager(
                activeNodeFields.values(), scalarVariables, elements, constraints, nodeSets, activeCells
            )

            dofVector = dofManager.constructDofVector()

        for acl in activeCells:
            dofVector[acl] += 10.0 * acl.cellNumber

        print("equation system would have a size of {:}".format(dofManager.nDof))

        shift = np.asarray([2.0, 3.0 * np.cos(4 * np.pi * i / 100.0)])
        mpmModel.advanceToTime(10.0 * (i))

        print("shifting all material points by {:}".format(shift))

        for mp in mpmModel.materialPoints.values():
            mp.addDisplacement(shift)

        activeNodeFields["displacement"]["dU"][:] = shift

        mpmModel.nodeFields["displacement"].copyEntriesFromOther(activeNodeFields["displacement"])

        fieldOutputController.finalizeIncrement()

        ensightOutput.finalizeIncrement()

        journal.printSeperationLine()

    ensightOutput.finalizeJob()
