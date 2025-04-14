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
from edelweissmpm.generators.rectangularparticlegridgenerator import (
    generateRectangularParticleGrid,
)
from edelweissmpm.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmpm.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmpm.meshfree.vci import (
    BoundaryParticleDefinition,
    VariationallyConsistentIntegrationManager,
)
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmpm.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmpm.particles.marmot.marmotparticlewrapper import MarmotParticleWrapper
from edelweissmpm.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmpm.solvers.nqsmarmotparallel import NQSParallelForMarmot


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
    length = 4
    nX = 40
    nY = 10
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
        "properties": np.array([30000.0, 0.3, 0.0, 1e-15, 2e-15, 1.4999, 1.0]),
    }

    def TheParticleFactory(number, coordinates, volume):
        return MarmotParticleWrapper(
            "GradientEnhancedMicropolar/PlaneStrain/Point", number, coordinates, volume, theApproximation, theMaterial
        )

    theModel = generateRectangularParticleGrid(
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
        "deformation gradient",
        theModel.particleSets["all"],
        "deformation gradient",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", theModel, fieldOutputController, theJournal, None)
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perNode")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perElement"
    )
    ensightOutput.initializeJob()

    dirichletLeft = ParticlePenaltyWeakDirichlet(
        "left", theModel, theModel.particleSets["rectangular_grid_left"], "displacement", {0: 0.0, 1: 0.0}, 1e6
    )
    dirichletRight = ParticlePenaltyWeakDirichlet(
        "right", theModel, theModel.particleSets["rectangular_grid_right"], "displacement", {0: 0, 1: 0.01}, 1e6
    )

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-1, 1e-1, 1e-1, 1000, theJournal)

    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)
    nonlinearSolver = NQSParallelForMarmot(theJournal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 3
    iterationOptions["allowed residual growths"] = 5

    linearSolver = pardisoSolve

    theBoundary = [
        BoundaryParticleDefinition(
            theModel.particleSets["rectangular_grid_left"], np.array([-1.0, 0.0]) * height / nY, 0
        ),
        BoundaryParticleDefinition(
            theModel.particleSets["rectangular_grid_right"], np.array([1.0, 0.0]) * height / nY, 0
        ),
        BoundaryParticleDefinition(
            theModel.particleSets["rectangular_grid_bottom"], np.array([0.0, -1.0]) * length / nX, 0
        ),
        BoundaryParticleDefinition(
            theModel.particleSets["rectangular_grid_top"], np.array([0.0, 1.0]) * length / nX, 0
        ),
    ]

    vciManager = VariationallyConsistentIntegrationManager(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values()), theBoundary
    )
    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=[dirichletLeft, dirichletRight],
            userIterationOptions=iterationOptions,
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

    return theModel


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def _test_sim():

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
