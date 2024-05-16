import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmpm.materialpoints.base.mp import MaterialPointBase
from edelweissmpm.models.mpmmodel import MPMModel


class PenaltyConstrainMP2Node(MPMConstraintBase):
    """
    This is a penalty constraint that constrains a material point to a node using a penalty method.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The full MPMModel instance.
    slaveMP
        The material point to constrain.
    masterNode
        The node to constrain to.
    field
        The field this constraint is acting on.
    prescribedComponents
        The index of the constrained component.
    penaltyParameter
        The penalty parameter value.
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        slaveMP: MaterialPointBase,
        masterNode: Node,
        field: str,
        prescribedComponents: list[int],
        penaltyParameter: float,
    ):
        self._name = name
        self._model = model
        self._slaveMP = slaveMP
        self._masterNode = masterNode
        self._field = field
        self._prescribedComponents = prescribedComponents
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._penaltyParameter = penaltyParameter
        self._slaveNodes = dict()

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return list(self._slaveNodes.keys()) + [self._masterNode]

    @property
    def fieldsOnNodes(self) -> list:
        return [
            [
                self._field,
            ]
        ] * len(self.nodes)

    @property
    def nDof(self) -> int:
        return len(self.nodes) * self._fieldSize

    @property
    def scalarVariables(
        self,
    ) -> list:
        return []

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def updateConnectivity(self, model: MPMModel):
        """Update the connectivity of the constraint.

        Parameters
        ----------
        model
            The full MPMModel instance.

        Returns
        -------
        bool
            True if the connectivity has changed, False otherwise.
        """

        nodes = {n: i for i, n in enumerate(set(n for c in self._slaveMP.assignedCells for n in c.nodes))}

        hasChanged = False
        if nodes != self._slaveNodes:
            hasChanged = True

        self._slaveNodes = nodes

        return hasChanged

    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):
        """Apply the penalty constraint to the residual and stiffness matrix.

        Parameters
        ----------
        dU
            The displacement vector.
        PExt
            The external force vector.
        V
            The stiffness matrix.
        timeStep
            The time step.
        """

        # loop over the prescribed components
        for i in self._prescribedComponents:

            P_i = PExt[i :: self._fieldSize]  # reshape the force vector accounting for the ith component
            dU_j = dU[i :: self._fieldSize]  # reshape the displacement vector accounting for the ith component

            K_ij = V.reshape((self.nDof, self.nDof))[i :: self._fieldSize, i :: self._fieldSize]

            masterNodeIndex = len(self._slaveNodes)  # index of the master node is the last one

            dU_Master_j = dU_j[masterNodeIndex]

            center = self._slaveMP.getCenterCoordinates()

            # loop over the cells of the slave material point
            for c in self._slaveMP.assignedCells:
                N = c.getInterpolationVector(center)

                slaveClNodeIdcs = [self._slaveNodes[n] for n in c.nodes]

                dU_MP_j = N @ dU_j[slaveClNodeIdcs]

                # potential to be minimized is:
                # psi = 0.5 * penalty * (dU_MP_j - dU_Master_j)^2

                P_i[slaveClNodeIdcs] += N * self._penaltyParameter * (dU_MP_j - dU_Master_j)
                P_i[masterNodeIndex] += self._penaltyParameter * (dU_MP_j - dU_Master_j) * -1

                K_ij[np.ix_(slaveClNodeIdcs, slaveClNodeIdcs)] += np.outer(N, N) * self._penaltyParameter
                K_ij[slaveClNodeIdcs, masterNodeIndex] += N * self._penaltyParameter * -1
                K_ij[np.ix_([masterNodeIndex], slaveClNodeIdcs)] += N * self._penaltyParameter * -1
                K_ij[masterNodeIndex, masterNodeIndex] += self._penaltyParameter
