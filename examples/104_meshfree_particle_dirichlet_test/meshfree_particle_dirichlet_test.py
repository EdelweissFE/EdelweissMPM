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

import numpy as np
import pytest


def run_sim():
    dimension = 2

    # set nump linewidth to 200:
    np.set_printoptions(linewidth=200)
    # set 2 digits after comma:
    np.set_printoptions(precision=2)
    # and let's print all the array:
    np.set_printoptions(threshold=np.inf)

    from edelweissfe.journal.journal import Journal

    theJournal = Journal()
    from edelweissmpm.models.mpmmodel import MPMModel

    theModel = MPMModel(dimension)

    from edelweissmpm.generators.rectangularkernelfunctiongridgenerator import (
        generateRectangularKernelFunctionGrid,
    )
    from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )

    def theMeshfreeKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=10.0)

    theModel = generateRectangularKernelFunctionGrid(
        theModel, theJournal, theMeshfreeKernelFunctionFactory, x0=-2, y0=-2, h=4, l=4, nX=3, nY=3
    )

    # let's define the type of approximation: We would like to have a reproducing kernel approximation of completeness order 1
    from edelweissmpm.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    # We need a dummy material for the material point
    theMaterial = {
        "material": "GMDamagedShearNeoHooke",
        "properties": np.array([30000.0, 0.3, 1, 1, 2, 1.4999]),
    }

    from edelweissmpm.particles.marmot.marmotparticlewrapper import (
        MarmotParticleWrapper,
    )

    def TheParticleFactory(number, coordinates, volume):
        return MarmotParticleWrapper(
            "GradientEnhancedMicropolar/PlaneStrain/Point", number, coordinates, volume, theApproximation, theMaterial
        )

    from edelweissmpm.generators.rectangularparticlegridgenerator import (
        generateRectangularParticleGrid,
    )

    theModel = generateRectangularParticleGrid(
        theModel, theJournal, TheParticleFactory, x0=-2, y0=-2, h=4, l=4, nX=3, nY=3
    )

    # for Semi-Lagrangian particle methods, we assoicate a particle with a kernel function.

    # let's create the particle kernel domain
    from edelweissmpm.meshfree.particlekerneldomain import ParticleKernelDomain

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )

    from edelweissmpm.particlemanagers.kdbinorganizedparticlemanager import (
        KDBinOrganizedParticleManager,
    )

    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True
    )

    # let's print some details
    print(theParticleManager)

    # We now create a bundled model.
    # We need this model to create the dof manager

    theModel.particleKernelDomains["my_all_with_all"] = theParticleKernelDomain

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController
    from edelweissmpm.numerics.dofmanager import MPMDofManager

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["all"],
        "displacement",
    )

    fieldOutputController.initializeJob()

    from edelweissmpm.outputmanagers.ensight import (
        OutputManager as EnsightOutputManager,
    )

    ensightOutput = EnsightOutputManager("ensight", theModel, fieldOutputController, theJournal, None)
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perNode")
    ensightOutput.initializeJob()

    nSteps = 12
    for i in range(nSteps):
        angle = 2 * np.pi * i / nSteps
        disp_x = 0.5 * np.cos(angle)
        disp_y = 0.5 * np.sin(angle)

        theParticleManager.updateConnectivity()

        theActiveSubModel = theModel.getActiveSubModel()

        theDofManager = MPMDofManager(
            theActiveSubModel.nodeFields.values(), particles=theActiveSubModel.particles.values()
        )

        theJournal.message(f"step: {i}", "info", 0)
        theJournal.message(f"number of active nodes: {len(theActiveSubModel.nodes)}", "info")

        U = theDofManager.constructDofVector()
        U[theDofManager.idcsOfNodeFieldsInDofVector["displacement"]][::dimension] = disp_x
        U[theDofManager.idcsOfNodeFieldsInDofVector["displacement"]][1::dimension] = disp_y

        for theParticle in theModel.particles.values():
            theParticleDisplacement = theParticle.getResultArray("displacement")[0:dimension]

            particleNodes = theParticle.nodes

            dUc = U[theDofManager.idcsOfParticlesInDofVector[theParticle]]

            sizeVec = theDofManager.nDof
            Pc = np.zeros(sizeVec)
            Kc = np.zeros((sizeVec * sizeVec))
            timeNew = 0.0
            dTime = 0.0

            theParticle.computePhysicsKernels(dUc, Pc, Kc, timeNew, dTime)

            theJournal.message(f"particle: {theParticle.number}", "info")
            theJournal.message(f"particle assigned nodes: {[n.label for n in particleNodes]}", "info")
            theJournal.message(
                f"particle indices in dof vector: {theDofManager.idcsOfParticlesInDofVector[theParticle]}", "info"
            )
            theJournal.message(f"applied incremental displacement: {disp_x}, {disp_y}", "info")
            theJournal.message(f"resulting total displacement: {theParticleDisplacement}", "info")

            N = theParticle.getInterpolationVector(theParticle.getCenterCoordinates())
            theJournal.message(f"interpolation vector: {N}", "info")

        theModel.advanceToTime(i)
        fieldOutputController.finalizeIncrement()
        ensightOutput.finalizeIncrement()

    fieldOutputController.finalizeJob()
    ensightOutput.finalizeJob()
    return Kc


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim():

    # disable plots and suppress warnings
    import matplotlib

    matplotlib.use("Agg")
    import warnings

    warnings.filterwarnings("ignore")

    lastStiffness = run_sim()

    gold = np.loadtxt("gold.csv")

    # assert np.isclose(lastStiffness, gold).all()
    assert np.isclose(np.linalg.norm(lastStiffness.flatten()), np.linalg.norm(gold.flatten()))


if __name__ == "__main__":
    lastStiffness = run_sim()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", lastStiffness)
