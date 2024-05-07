"""
Constraints are used to enforce certain conditions on the system. They are used to enforce boundary conditions, contact conditions, etc.
Technically, constraints are implemented as a set of kernels, which are applied to the system matrix and the external load vector.
In this regard, constraints are similar to elements, but they mey also use additional scalar variables to enforce the conditions.
Unlike elements, they are not subject to external loads.
"""

from abc import ABC, abstractmethod

import numpy as np
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.models.mpmmodel import MPMModel


class MPMConstraintBase(ABC):
    """The MPMConstraintBase class is an abstract base class for all constraints.
    If you want to implement a new constraint, you have to inherit from this class."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this constraint.

        Returns
        -------
        str
            The name."""
        pass

    @property
    @abstractmethod
    def nodes(self) -> list[Node]:
        """The nodes this constraint is acting on.
        Duplicates are _allowed_.

        Returns
        -------
        list[Node]
            The list of nodes."""
        pass

    @property
    @abstractmethod
    def fieldsOnNodes(self) -> list[list[str]]:
        """The fields on the nodes this constraint is acting on.

        Returns
        -------
        list[list[str]]
            The node-wise list of fields."""
        pass

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The total number of degrees of freedom this constraint is associated with.

        Returns
        -------
        int
            The total number of degrees of freedom."""
        pass

    @property
    @abstractmethod
    def scalarVariables(
        self,
    ) -> list[ScalarVariable]:
        """The list of assigned ScalarVariable.

        Returns
        -------
        list[ScalarVariable]
            The list of scalar variables."""
        pass

    @abstractmethod
    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        """Tell the framework how many scalar variables (e.g., Lagrangian multipliers)
        this constraint needs.

        Returns
        -------
        int
            The number of requested ScalarVariable
        """
        pass

    @abstractmethod
    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        """This is the list of constraint specific scalar variables which are assigned to this constraint.

        Parameters
        ----------
        scalarVariables
            The list of ScalarVariable to be assigned.
        """
        pass

    @abstractmethod
    def updateConnectivity(self, model: MPMModel) -> bool:
        """This method is called before each new timeStep, after material point connectivity was updated, but before the global equation system is created.
        If the contribution to the global system changes, True is returned.

        Parameters
        ----------
        model
            The current model.

        Returns
        -------
        bool
            The truth value if the connectivity has changed.
        """
        pass

    @abstractmethod
    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):
        """Apply the constraint, i.e., compute the 'kernels'. Add the contributions to the external load vector and the system matrix.

        Parameters
        ----------
        dU
            The current increment since the last time the constraint was applied.
        PExt
            The external load vector.
        V
            The system (stiffness) matrix.
        timeStep
            The current step and total time.
        """
        pass
