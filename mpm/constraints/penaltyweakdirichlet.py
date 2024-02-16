from fe.config.phenomena import getFieldSize
from fe.variables.scalarvariable import ScalarVariable
from fe.timesteppers.timestep import TimeStep
from mpm.models.mpmmodel import MPMModel
from mpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from mpm.materialpoints.base.mp import MaterialPointBase
import numpy as np


class PenaltyWeakDirichlet(MPMConstraintBase):
    """
    This is an implementation of weak Dirichlet boundary conditions using a penalty formulation.
    It constrains a material point field increment.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The full MPMModel instance.
    constrainedMaterialPoints
        The list of the material point to be constrained.
    field
        The field this constraint is acting on.
    prescribedStepDelta
        The dictionary containing the prescribed bc components for the field in the present load step.
    penaltyParameter
        The penalty parameter value.
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        constrainedMaterialPoints: list[MaterialPointBase],
        field: str,
        prescribedStepDelta: dict,
        penaltyParameter: float,
    ):
        self._name = name
        self._model = model
        self._constrainedMPs = constrainedMaterialPoints
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._penaltyParameter = penaltyParameter
        self._nodes = dict()

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes.keys()

    @property
    def fieldsOnNodes(self) -> list:
        return [
            [
                self._field,
            ]
        ] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

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

    def updateConnectivity(self, model):
        nodes = {
            n: i for i, n in enumerate(set(n for mp in self._constrainedMPs for c in mp.assignedCells for n in c.nodes))
        }

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        return hasChanged

    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):
        for i, prescribedComponent in self._prescribedStepDelta.items():
            P_i = PExt[i :: self._fieldSize]
            dU_j = dU[i :: self._fieldSize]

            K_ij = V.reshape((self.nDof, self.nDof))[i :: self._fieldSize, i :: self._fieldSize]

            for mp in self._constrainedMPs:
                center = mp.getCenterCoordinates()

                for c in mp.assignedCells:
                    N = c.getInterpolationVector(center)

                    nodeIdcs = [self._nodes[n] for n in c.nodes]

                    mpValue = N @ dU_j[nodeIdcs]

                    P_i[nodeIdcs] += (
                        N * self._penaltyParameter * (mpValue - prescribedComponent * timeStep.stepProgressIncrement)
                    )
                    K_ij[np.ix_(nodeIdcs, nodeIdcs)] += np.outer(N, N) * self._penaltyParameter
