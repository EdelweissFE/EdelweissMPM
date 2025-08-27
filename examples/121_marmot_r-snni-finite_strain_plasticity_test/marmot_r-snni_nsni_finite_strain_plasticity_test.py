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

from edelweissmpm.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmpm.generators.rectangularkernelfunctiongridgenerator import (
    generateRectangularKernelFunctionGrid,
)
from edelweissmpm.generators.rectangularquadparticlegridgenerator import (
    generateRectangularQuadParticleGrid,
)
from edelweissmpm.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmpm.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmpm.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmpm.particles.marmot.marmotparticlewrapper import MarmotParticleWrapper
from edelweissmpm.solvers.nqs import NonlinearQuasistaticSolver

# from edelweissmpm.generators.rectangularparticlegridgenerator import (
#     generateRectangularParticleGrid,
# )


def run_sim(no_limit=False):
    dimension = 2

    E = 20000
    nu = 0.3
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))

    timeScalingFactor = 2e-5
    massDensityScalingFactor = 1.0 / (timeScalingFactor**2)

    vProjectileInitial = -200 * 1e3 * timeScalingFactor

    particleSize = 1.0 / 16
    supportRadius = particleSize * 1.7

    x0Plate = 0
    y0Plate = 0
    heightPlate = 0.5
    lengthPlate = 5.0
    nXPlate = int(lengthPlate / particleSize)
    nYPlate = int(heightPlate / particleSize)

    heightProjectile = 0.5
    lengthProjectile = 0.25
    nXProjectile = int(lengthProjectile / particleSize)
    nYProjectile = int(heightProjectile / particleSize)
    # place projectile above plate, centered in x direction
    x0Projectile = (lengthPlate - lengthProjectile) / 2
    y0Projectile = 1.5

    # set nump linewidth to 200:
    np.set_printoptions(linewidth=200)
    # set 2 digits after comma:
    np.set_printoptions(precision=2)
    # and let's print all the array:
    np.set_printoptions(threshold=np.inf)

    theJournal = Journal()

    theModel = MPMModel(dimension)

    def theMeshfreeKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=3)

    theModel = generateRectangularKernelFunctionGrid(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        x0=x0Plate,
        y0=y0Plate,
        h=heightPlate,
        l=lengthPlate,
        nX=nXPlate,
        nY=nYPlate,
        name="plate",
    )

    theModel = generateRectangularKernelFunctionGrid(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        x0=x0Projectile,
        y0=y0Projectile,
        h=heightProjectile,
        l=lengthProjectile,
        nX=nXProjectile,
        nY=nYProjectile,
        name="projectile",
        firstKernelFunctionNumber=nXPlate * nYPlate + 1,
    )

    # let's define the type of approximation: We would like to have a reproducing kernel approximation of completeness order 1
    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    theMaterial = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K, G, 1e1, 1e-1, 1e-0, 0, 1, 1e-9 * massDensityScalingFactor]),
    }

    theMaterialProjectile = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K * 10, G * 10, 20e10, 20e10, 0, 0, 1, 1e-8 * massDensityScalingFactor]),
    }

    def ThePlateFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "DisplacementSQCNI_RxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            volume,
            theApproximation,
            theMaterial,
        )

    def TheProjectileFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "DisplacementSQCNI_RxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            volume,
            theApproximation,
            theMaterialProjectile,
        )

    theModel = generateRectangularQuadParticleGrid(
        theModel,
        theJournal,
        ThePlateFactory,
        x0=x0Plate,
        y0=y0Plate,
        h=heightPlate,
        l=lengthPlate,
        nX=nXPlate,
        nY=nYPlate,
        name="plate",
    )
    theModel = generateRectangularQuadParticleGrid(
        theModel,
        theJournal,
        TheProjectileFactory,
        x0=x0Projectile,
        y0=y0Projectile,
        h=heightProjectile,
        l=lengthProjectile,
        nX=nXProjectile,
        nY=nYProjectile,
        firstParticleNumber=nXPlate * nYPlate + 1,
        name="projectile",
    )

    for particle in theModel.particles.values():
        particle.setProperty("newmark-beta beta", 0.28)
        particle.setProperty("newmark-beta gamma", 0.53)

    # let's create the particle kernel domain
    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )

    # for Semi-Lagrangian particle methods, we assoicate a particle with a kernel function.
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

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["projectile_all"],
        "displacement",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "velocity",
        theModel.particleSets["all"],
        "velocity",
    )

    fieldOutputController.addPerParticleFieldOutput(
        "plate acceleration",
        theModel.particleSets["plate_all"],
        "acceleration",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "particle acceleration",
        theModel.particleSets["all"],
        "acceleration",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "plate velocity",
        theModel.particleSets["plate_all"],
        "velocity",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "particle velocity",
        theModel.particleSets["all"],
        "velocity",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "plate vertex displacements",
        theModel.particleSets["plate_all"],
        "vertex displacements",
        f_x=lambda x: np.pad(np.reshape(x, (-1, 2)), ((0, 0), (0, 1)), mode="constant", constant_values=0),
    )
    fieldOutputController.addPerParticleFieldOutput(
        "projectile vertex displacements",
        theModel.particleSets["projectile_all"],
        "vertex displacements",
        f_x=lambda x: np.pad(np.reshape(x, (-1, 2)), ((0, 0), (0, 1)), mode="constant", constant_values=0),
    )
    fieldOutputController.addPerParticleFieldOutput(
        "deformation gradient",
        theModel.particleSets["all"],
        "deformation gradient",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", theModel, fieldOutputController, theJournal, None)
    ensightOutput.minDTForOutput = 1e-3
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["plate acceleration"], create="perElement", name="acceleration"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["particle acceleration"],
        create="perElement",
        name="acceleration",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["plate velocity"], create="perElement", name="velocity"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["particle velocity"], create="perElement", name="velocity"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["plate vertex displacements"],
        create="perNode",
        name="vertex displacements",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["projectile vertex displacements"],
        create="perNode",
        name="vertex displacements",
    )
    ensightOutput.initializeJob()

    dirichletLeft = ParticlePenaltyWeakDirichlet(
        "left", theModel, theModel.particleSets["plate_left"], "displacement", {0: 0, 1: 0}, 1e6
    )

    dirichletRight = ParticlePenaltyWeakDirichlet(
        "right", theModel, theModel.particleSets["plate_right"], "displacement", {0: 0, 1: 0.0}, 1e6
    )

    dirichlets = [
        dirichletLeft,
        dirichletRight,
    ]

    incSize = 5e-3
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0, 1.0, incSize, incSize, incSize / 1e8, 50 if not no_limit else 10000, theJournal, increaseFactor=1.5
    )

    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 10
    iterationOptions["critical iterations"] = 7
    iterationOptions["allowed residual growths"] = 2
    iterationOptions["default relative flux residual tolerance"] = 1e-3
    iterationOptions["default absolute flux residual tolerance"] = 1e-12
    iterationOptions["default absolute field correction tolerance"] = 1e-9
    iterationOptions["fall back to quasi Newton after n residual growths"] = False

    linearSolver = pardisoSolve

    for p in theModel.particleSets["projectile_all"]:
        v = p.getResultArray("velocity")
        v[1] = vProjectileInitial
        p.acceptStateAndPosition()

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=dirichlets,
            userIterationOptions=iterationOptions,
        )

    except StepFailed as e:
        theJournal.message(f"Step failed: {str(e)}", "error")
        if not no_limit:
            theJournal.message(
                "This is an expected behaviour for this test. Rerun with --no-limit to run until the end.", "error"
            )

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = theJournal.linewidth
        theJournal.printPrettyTable(prettytable, "Summary")

    return theModel, fieldOutputController


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

    theModel, fieldOutputController = run_sim()

    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    gold = np.loadtxt("gold.csv")

    assert np.isclose(np.copy(res.flatten() - gold.flatten()), 0.0, rtol=1e-12).all()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    parser.add_argument(
        "--no-limit", dest="no_limit", action="store_true", help="do not limit the number of time increments."
    )
    args = parser.parse_args()

    theModel, fieldOutputController = run_sim(args.no_limit)
    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    if args.create_gold:
        np.savetxt("gold.csv", res.flatten())
