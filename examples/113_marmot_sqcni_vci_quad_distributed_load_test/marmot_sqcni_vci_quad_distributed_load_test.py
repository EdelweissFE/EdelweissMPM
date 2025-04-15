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


def run_sim():
    dimension = 2

    # set nump linewidth to 200:
    np.set_printoptions(linewidth=200)
    # set 2 digits after comma:
    np.set_printoptions(precision=2)
    # and let's print all the array:
    np.set_printoptions(threshold=np.inf)

    theJournal = Journal()

    theModel = MPMModel(dimension)

    x0 = -1
    y0 = -1
    height = 1
    length = 8
    nX = 8 * 4
    nY = 1 * 4
    supportRadius = 0.5

    def theMeshfreeKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=2)

    theModel = generateRectangularKernelFunctionGrid(
        theModel, theJournal, theMeshfreeKernelFunctionFactory, x0=x0, y0=y0, h=height, l=length, nX=nX, nY=nY
    )

    # let's define the type of approximation: We would like to have a reproducing kernel approximation of completeness order 1
    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    # We need a dummy material for the material point
    theMaterial = {
        "material": "GMDamagedShearNeoHooke",
        "properties": np.array([3000.0, 0.2, 1, 0.1, 0.2, 1.4999, 1.0]),
    }

    def TheParticleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "GradientEnhancedMicropolarSQCNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            volume,
            theApproximation,
            theMaterial,
        )

    theModel = generateRectangularQuadParticleGrid(
        theModel, theJournal, TheParticleFactory, x0=x0, y0=y0, h=height, l=length, nX=nX, nY=nY
    )

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
        theModel.particleSets["all"],
        "displacement",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements",
        theModel.particleSets["all"],
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
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        create="perNode",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perElement"
    )
    ensightOutput.initializeJob()

    dirichletLeft = ParticlePenaltyWeakDirichlet(
        "left", theModel, theModel.particleSets["rectangular_grid_left"], "displacement", {0: 0, 1: 0}, 1e6
    )

    dirichlets = [
        dirichletLeft,
    ]

    incSize = 1e-0
    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, incSize, incSize, incSize / 1e3, 20, theJournal)

    # nonlinearSolver = NQSParallelForMarmot(theJournal)
    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 3
    iterationOptions["allowed residual growths"] = 10

    linearSolver = pardisoSolve

    from edelweissmpm.meshfree.vci import (
        BoundaryParticleDefinition,
        VariationallyConsistentIntegrationManager,
    )

    theBoundary = [
        BoundaryParticleDefinition(theModel.particleSets["rectangular_grid_left"], np.empty(2), 4),
        BoundaryParticleDefinition(theModel.particleSets["rectangular_grid_right"], np.empty(2), 2),
        BoundaryParticleDefinition(theModel.particleSets["rectangular_grid_bottom"], np.empty(2), 1),
        BoundaryParticleDefinition(theModel.particleSets["rectangular_grid_top"], np.empty(2), 3),
    ]

    vciManager = VariationallyConsistentIntegrationManager(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values()), theBoundary
    )

    from edelweissmpm.stepactions.particledistributedload import ParticleDistributedLoad

    pressureTop = ParticleDistributedLoad(
        "pressureTop",
        theModel,
        theJournal,
        theModel.particleSets["rectangular_grid_top"],
        "pressure",
        np.array([-2]),
        surfaceID=3,
    )

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
            particleDistributedLoads=[pressureTop],
            vciManagers=[vciManager],
        )

    except StepFailed as e:
        theJournal.message(f"Step failed: {str(e)}", "error")
        raise

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

    assert np.isclose(res.flatten(), gold.flatten(), rtol=1e-12).all()


if __name__ == "__main__":
    theModel, fieldOutputController = run_sim()
    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", res.flatten())
