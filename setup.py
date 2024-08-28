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
from setuptools import find_packages, setup
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


class MarmotExtension(Extension):
    """A custom extension that links against Marmot."""

    def __init__(self, pyxpath):
        super().__init__(
            "*",
            sources=[
                pyxpath,
            ],
            include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
            libraries=["Marmot"],
            library_dirs=[join(marmot_dir, "lib")],
            runtime_library_dirs=[join(marmot_dir, "lib")],
            language="c++",
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
    MarmotExtension("edelweissmpm/materialpoints/marmotmaterialpoint.pyx"),
]

extensions += [
    Extension(
        "*",
        sources=[
            "edelweissmpm/solvers/nqsmarmotparallel.pyx",
        ],
        include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
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


extensions += [
    MarmotExtension("edelweissmpm/particles/marmot/marmotparticle.pyx"),
]

extensions += [MarmotExtension("edelweissmpm/meshfreeshapefunctions/marmot/marmotreproducingkernelshapefunction.pyx")]

setup(
    name="EdelweissMPM",
    version="v24.04",
    description="EdelweissMPM: A material point solver module for EdelweissFE.",
    license="LGPL-2.1",
    packages=find_packages(),
    include_package_data=True,
    author="Matthias Neuner",
    author_email="matthias.neuner@uibk.ac.at",
    url="https://github.com/EdelweissFE/EdelweissMPM",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, compiler_directives=directives, annotate=True, language_level=3),
),


print("Finish!")
