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

from edelweissmpm.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)


def run_sim():
    dimension = 2

    from edelweissfe.points.node import Node

    from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )

    # create 4 kernel functions

    supportRadii = [2.0, 2.0, 2.0, 2.0]

    # we set all node coordinates to zero. In principle, we could set them to any value here,
    # but we will move the kernel functions later on to different positions to demonsrate the moving of kernel functions.
    testNode1 = Node(1, np.array([0.0, 0.0]))
    kernelFunction1 = MarmotMeshfreeKernelFunctionWrapper(testNode1, "BSplineBoxed", supportRadius=supportRadii[0])

    testNode2 = Node(2, np.array([0.00, 0.0]))
    kernelFunction2 = MarmotMeshfreeKernelFunctionWrapper(testNode2, "BSplineBoxed", supportRadius=supportRadii[1])
    kernelFunction2.move(np.array([1.0, 0.0]))

    testNode3 = Node(3, np.array([0.0, 0.0]))
    kernelFunction3 = MarmotMeshfreeKernelFunctionWrapper(testNode3, "BSplineBoxed", supportRadius=supportRadii[2])
    kernelFunction3.move(np.array([1.0, 1.0]))

    testNode4 = Node(4, np.array([0.0, 0.0]))
    kernelFunction4 = MarmotMeshfreeKernelFunctionWrapper(testNode4, "BSplineBoxed", supportRadius=supportRadii[3])
    kernelFunction4.move(np.array([0.0, 1.0]))

    # let's define the type of approximation: We would like to have a reproducing kernel approximation of completeness order 1
    from edelweissmpm.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    # We need a dummy material for the mateiral point
    theMaterial = {
        "material": "GMDamagedShearNeoHooke",
        "properties": np.array([30000.0, 0.3, 1, 1, 2, 1.4999]),
    }
    mpCoordinates = np.array([-0.0, 0.0]).reshape(-1, dimension)
    mpVolume = 1.0

    # let's instance the material point
    from edelweissmpm.materialpoints.marmotmaterialpoint.mp import (
        MarmotMaterialPointWrapper,
    )

    mp = MarmotMaterialPointWrapper("GradientEnhancedMicropolar/PlaneStrain", 1, mpCoordinates, mpVolume, theMaterial)

    # and finally .. create the particle. The particle hosts the material point, which again hosts the material.
    from edelweissmpm.particles.marmot.marmotparticlewrapper import (
        MarmotParticleWrapper,
    )

    marmotParticle1 = MarmotParticleWrapper(
        "GradientEnhancedMicropolar/PlaneStrain/Point", 1, mpCoordinates, mpVolume, mp, theApproximation
    )

    # create the particle manager
    # The particle manager is responsible for the organization of the kernel functions and the particles.
    # In detail, it determines the connectivity of the kernel functions and the particles, and it assigns all covering kernel functions to the particles.
    theParticleManager = KDBinOrganizedParticleManager(
        [kernelFunction1, kernelFunction2, kernelFunction3, kernelFunction4], [marmotParticle1], dimension
    )
    # let's print some details
    print(theParticleManager)

    # let the manager do its job
    theParticleManager.updateConnectivity()

    # now let's move the particle and update the connectivity
    # to this end, we get access to the particle's displacement (this is usually the result of a simulation step)
    mpDisplacementView = mp.getResultArray("displacement")

    # # let's move the particle in a circle
    # # and make a nice animation
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="datalim")

    # first we illustrate the support of the kernel functions
    theKernelFunctions = [kernelFunction1, kernelFunction2, kernelFunction3, kernelFunction4]
    for i in range(4):
        theKernelFunction = theKernelFunctions[i]

        color = ["r", "g", "b", "y"]

        boundingBox = theKernelFunction.getBoundingBox()
        ax.add_patch(
            plt.Rectangle(
                boundingBox[0],
                boundingBox[1][0] - boundingBox[0][0],
                boundingBox[1][1] - boundingBox[0][1],
                color=color[i],
                fill=True,
                alpha=0.5,
            )
        )

    theParticle = ax.plot(
        marmotParticle1.getVertexCoordinates()[0][0], marmotParticle1.getVertexCoordinates()[0][1], "ro"
    )[0]

    nAssignedKernelFunctions = len(marmotParticle1.getAssignedKernelFunctions())
    nKernelFunctionsAnnotation = ax.annotate(
        str(nAssignedKernelFunctions),
        (marmotParticle1.getVertexCoordinates()[0][0], marmotParticle1.getVertexCoordinates()[0][1]),
    )

    currentCoordinates = marmotParticle1.getVertexCoordinates()[0:dimension].flatten()
    binIdx = theParticleManager._theBins._getBinIndices(currentCoordinates)
    nFunctionsInBin = len(theParticleManager._theBins._thebins[binIdx[0], binIdx[1]])

    binIdxAnnotation = ax.annotate("bin indices: " + str(binIdx), (-1, -1))
    nFunctionsInBinAnnotation = ax.annotate("n functions in bin: " + str(nFunctionsInBin), (-1, -2))
    currentCoordinatesAnnotation = ax.annotate("current coordinates: " + str(currentCoordinates), (-1, -3))

    nFrames = 100
    coveredDomainMin, coveredDomainMax = theParticleManager.getCoveredDomain()

    pathx = (
        np.cos(np.linspace(0, 2 * np.pi, nFrames)) * 0.75 * (coveredDomainMax[0] - coveredDomainMin[0]) / 2
        + (coveredDomainMax[0] + coveredDomainMin[0]) / 2
    )
    pathy = (
        np.sin(np.linspace(0, 2 * np.pi, nFrames)) * 0.75 * (coveredDomainMax[1] - coveredDomainMin[1]) / 2
        + (coveredDomainMax[1] + coveredDomainMin[1]) / 2
    )

    def update(frame):
        # animate theParticle position:
        mpDisplacementView[0] = pathx[frame]
        mpDisplacementView[1] = pathy[frame]
        theParticleManager.updateConnectivity()

        currentCoordinates = marmotParticle1.getVertexCoordinates()[0:dimension].flatten()

        currentCoordinatesAnnotation.set_text("current coordinates: " + str(currentCoordinates))

        theParticle.set_data((currentCoordinates[0],), (currentCoordinates[1],))
        nKernelFunctionsAnnotation.set_position((currentCoordinates[0] + 0.1, currentCoordinates[1]))
        nKernelFunctionsAnnotation.set_text(
            "n assigned kernel functions: " + str(len(marmotParticle1.getAssignedKernelFunctions()))
        )

        theBinIndices = theParticleManager._theBins._getBinIndices(currentCoordinates)
        binIdxAnnotation.set_text("bin indices: " + str(theBinIndices))
        binIdxAnnotation.set_position((currentCoordinates[0] + 0.1, currentCoordinates[1] - 0.2))

        nFunctionsInBin = len(theParticleManager._theBins._thebins[theBinIndices[0], theBinIndices[1]])
        nFunctionsInBinAnnotation.set_text("n functions in bin: " + str(nFunctionsInBin))
        nFunctionsInBinAnnotation.set_position((currentCoordinates[0] + 0.1, currentCoordinates[1] - 0.4))

        return [
            theParticle,
            nKernelFunctionsAnnotation,
            binIdxAnnotation,
            nFunctionsInBinAnnotation,
            currentCoordinatesAnnotation,
        ]

    # set limits to the bounding boxes min and max of kernel functions:
    boundingBoxes = [kf.getBoundingBox() for kf in theKernelFunctions]
    minx = min([bb[0][0] for bb in boundingBoxes])
    maxx = max([bb[1][0] for bb in boundingBoxes])
    miny = min([bb[0][1] for bb in boundingBoxes])
    maxy = max([bb[1][1] for bb in boundingBoxes])

    ax.set_xlim(minx - 1, maxx + 1)
    ax.set_ylim(miny - 1, maxy + 1)

    anim = animation.FuncAnimation(fig=fig, func=update, frames=nFrames, interval=100)
    anim.save("animation.gif", writer="imagemagick", fps=10)

    plt.show()

    # let's illustrate the bin fill status
    theBins = theParticleManager._theBins._thebins

    theBinFillstatus = np.zeros_like(theBins, dtype=int)

    for i in range(theBins.shape[0]):
        for j in range(theBins.shape[1]):
            theBinFillstatus[i, j] = len(theBins[i, j])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cax = ax.matshow(theBinFillstatus, cmap="viridis")

    for i in range(theBins.shape[0]):
        for j in range(theBins.shape[1]):
            ax.text(j, i, theBinFillstatus[i, j], ha="center", va="center", color="black")

    fig.colorbar(cax)

    plt.show()

    # Finally, let's visualize the shape functions for a certain domain

    import matplotlib.pyplot as plt

    x = np.linspace(-0.99, 1.99, 100)
    y = np.linspace(-0.99, 1.99, 100)
    X, Y = np.meshgrid(x, y)

    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    Z3 = np.zeros_like(X)
    Z4 = np.zeros_like(X)
    # also for the gradients
    Z1_x = np.zeros_like(X)
    Z2_x = np.zeros_like(X)
    Z3_x = np.zeros_like(X)
    Z4_x = np.zeros_like(X)
    Z1_y = np.zeros_like(X)
    Z2_y = np.zeros_like(X)
    Z3_y = np.zeros_like(X)
    Z4_y = np.zeros_like(X)

    for i in range(100):
        for j in range(100):
            testpoint = np.array([X[i, j], Y[i, j]])
            # theShapeFunctions = theApproximation.computeShapeFunctions(
            #     testpoint, [kernelFunction1, kernelFunction2, kernelFunction3, kernelFunction4]
            # )
            theShapeFunctions, theShapeFunctionsGradients = theApproximation.computeShapeFunctionsAndGradients(
                testpoint, [kernelFunction1, kernelFunction2, kernelFunction3, kernelFunction4]
            )
            Z1[i, j] = theShapeFunctions[0]
            Z2[i, j] = theShapeFunctions[1]
            Z3[i, j] = theShapeFunctions[2]
            Z4[i, j] = theShapeFunctions[3]
            Z1_x[i, j] = theShapeFunctionsGradients[0, 0]
            Z2_x[i, j] = theShapeFunctionsGradients[1, 0]
            Z3_x[i, j] = theShapeFunctionsGradients[2, 0]
            Z4_x[i, j] = theShapeFunctionsGradients[3, 0]
            Z1_y[i, j] = theShapeFunctionsGradients[0, 1]
            Z2_y[i, j] = theShapeFunctionsGradients[1, 1]
            Z3_y[i, j] = theShapeFunctionsGradients[2, 1]
            Z4_y[i, j] = theShapeFunctionsGradients[3, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z1, label="Shape function 1")
    ax.plot_surface(X, Y, Z2, label="Shape function 2")
    ax.plot_surface(X, Y, Z3, label="Shape function 3")
    ax.plot_surface(X, Y, Z4, label="Shape function 4")
    ax.plot_surface(X, Y, Z1 + Z2 + Z3 + Z4, label="Sum of shape functions")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z1_x, label="Shape function 1 x gradient")
    ax.plot_surface(X, Y, Z2_x, label="Shape function 2 x gradient")
    ax.plot_surface(X, Y, Z3_x, label="Shape function 3 x gradient")
    ax.plot_surface(X, Y, Z4_x, label="Shape function 4 x gradient")
    ax.plot_surface(X, Y, Z1_x + Z2_x + Z3_x + Z4_x, label="Sum of shape functions x gradients")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z1_y, label="Shape function 1 y gradient")
    ax.plot_surface(X, Y, Z2_y, label="Shape function 2 y gradient")
    ax.plot_surface(X, Y, Z3_y, label="Shape function 3 y gradient")
    ax.plot_surface(X, Y, Z4_y, label="Shape function 4 y gradient")
    ax.plot_surface(X, Y, Z1_y + Z2_y + Z3_y + Z4_y, label="Sum of shape functions y gradients")

    plt.show()

    return np.array([Z1, Z2, Z3, Z4]).ravel()


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

    shapeFunctionValues = run_sim()

    gold = np.loadtxt("gold.csv")

    assert np.isclose(shapeFunctionValues, gold).all()


if __name__ == "__main__":
    shapeFunctionValues = run_sim()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", shapeFunctionValues)
