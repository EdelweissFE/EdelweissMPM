Welcome to EdelweissMPM
=======================

EdelweissMPM is a light-weight, platform-independent, parallel mateiral point method framework for the simulation of coupled problems in solid mechanics.
It is based on `EdelweissFE <https://github.com/EdelweissFE/EdelweissFE/>`_, which is a hard dependency.
Furthermore, by default, it makes use of the `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_ library for cell, material point, and material formulations.

EdelweissMPM aims to be...

 - ... a development and learning environment for constitutive models and the material point method,
 - ... an easy to use tool for coupled problems,
 - ... a very flexible tool for implementing and employing special techniques (e.g., the indirect displacement control technique),
   which are often more difficult and time consuming to implement in mature, MPI-parallelized codes,
 - ... an efficient tool for nonlinear simulations up to medium sized problems (~ :math:`10^5` degrees of freedom).

.. figure:: borehole.png
   :align: center
   :width: 240px

.. toctree::
   :maxdepth: 2
   :hidden:

   documentation/index

Execute a simulation
********************

Unline EdelweissFE, EdelweissMPM does not make use of input files.
All simulations are directly implemented in Python scripts.

Run a simulation simply by calling

.. code-block:: console

    python your_simulation.py

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
