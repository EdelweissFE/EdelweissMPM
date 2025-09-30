<!-- [![documentation](https://github.com/EdelweissMPM/EdelweissMPM/actions/workflows/sphinx.yml/badge.svg)](https://edelweissfe.github.io/EdelweissMPM) -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# EdelweissMPM: A light-weight, platform-independent, parallel material point module for EdelweissFE.

<!-- <p align="center"> -->
<!--   <img width="512" height="512" src="./doc/source/borehole_damage_lowdilation.gif"> -->
<!-- </p> -->

<!-- See the [documentation](https://edelweissfe.github.io/EdelweissMPM). -->

EdelweissMPM aims at an easy to understand, yet efficient implementation of the material point method.
Some features are:

 * Python for non performance-critical routines
 * Cython for performance-critical routines
 * Parallelization
 * Modular system, which is easy to extend
 * Output to Paraview, Ensight, CSV, matplotlib
 * Interfaces to powerful direct and iterative linear solvers

EdelweissMPM makes use of the [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/) library for cells, material points and constitutive model formulations.

Please note that the current public EdelweissMPM requires Marmot cells and material points for being able to run simulations.
Marmot cells and material points are currently not open source. If you are interested in using them, please reach out to us.
