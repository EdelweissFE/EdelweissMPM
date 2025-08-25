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
import os
import sys
from os.path import expanduser, join

import numpy
from Cython.Build import build_ext, cythonize
from setuptools import setup
from setuptools.extension import Extension

directives = {
    "boundscheck": False,
    "wraparound": False,
    "nonecheck": False,
    "initializedcheck": False,
}

default_install_prefix = sys.prefix
print("*" * 80)
print("EdelweissMPM setup")
print("System prefix: " + sys.prefix)
print("*" * 80)

marmot_dir = expanduser(os.environ.get("MARMOT_INSTALL_DIR", default_install_prefix))
print("Marmot install directory (overwrite via environment var. MARMOT_INSTALL_DIR):")
print(marmot_dir)
print("*" * 80)

extensions = list()


def MarmotExtension(pyxpath, *args, **kwargs):
    """A custom extension that links against Marmot."""

    return Extension(
        "*",
        sources=[
            pyxpath,
        ],
        include_dirs=[join(marmot_dir, "include"), join(marmot_dir, "include", "eigen3"), numpy.get_include()],
        libraries=["Marmot"],
        library_dirs=[join(marmot_dir, "lib")],
        runtime_library_dirs=[join(marmot_dir, "lib")],
        language="c++",
        *args,
        **kwargs,
    )


extensions += [
    MarmotExtension("edelweissmpm/cells/marmotcell/marmotcell.pyx"),
]

extensions += [
    MarmotExtension("edelweissmpm/cells/marmotcell/lagrangianmarmotcell.pyx"),
]

extensions += [
    MarmotExtension("edelweissmpm/cells/marmotcell/bsplinemarmotcell.pyx"),
]

extensions += [
    Extension(
        "*",
        sources=[
            "edelweissmpm/mpmmanagers/utils.pyx",
        ],
        include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
        runtime_library_dirs=[join(marmot_dir, "lib")],
        language="c++",
    )
]

extensions += [
    Extension(
        "*",
        sources=[
            "edelweissmpm/fieldoutput/mpresultcollector.pyx",
        ],
        include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
        runtime_library_dirs=[join(marmot_dir, "lib")],
        language="c++",
    )
]

extensions += [
    MarmotExtension("edelweissmpm/materialpoints/marmotmaterialpoint/mp.pyx"),
]

extensions += [
    Extension(
        "*",
        sources=[
            "edelweissmpm/solvers/base/parallelization.pyx",
        ],
        include_dirs=[join(marmot_dir, "include"), join(marmot_dir, "include", "eigen3"), numpy.get_include()],
        language="c++",
        extra_compile_args=[
            "-fopenmp",
            "-Wno-maybe-uninitialized",
        ],
        extra_link_args=["-fopenmp"],
    )
]

extensions += [
    MarmotExtension("edelweissmpm/cellelements/marmotcellelement/marmotcellelement.pyx"),
]

extensions += [
    MarmotExtension("edelweissmpm/cellelements/marmotcellelement/lagrangianmarmotcellelement.pyx"),
]


extensions += [MarmotExtension("edelweissmpm/meshfree/kernelfunctions/marmot/marmotmeshfreekernelfunction.pyx")]
extensions += [MarmotExtension("edelweissmpm/meshfree/approximations/marmot/marmotmeshfreeapproximation.pyx")]
extensions += [
    MarmotExtension("edelweissmpm/particles/marmot/marmotparticlewrapper.pyx"),
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, compiler_directives=directives, annotate=True, language_level=3),
),
