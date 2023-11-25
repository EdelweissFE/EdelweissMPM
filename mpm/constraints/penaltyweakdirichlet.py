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
    constrainedMaterialPoint
        The instance of the material point to be constrained.
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
        constrainedMaterialPoint: MaterialPointBase,
        field: str,
        prescribedStepDelta: dict,
        penaltyParameter: float,
    ):
        self._name = name
        self._model = model
        self._constrainedMP = constrainedMaterialPoint
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._penaltyParameter = penaltyParameter

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

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

    def initializeTimeStep(self, model, timeStep):
        self._nodes = [n for c in self._constrainedMP.assignedCells for n in c.nodes]

    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):
        idxP = 0
        idxK = 0

        for c in self._constrainedMP.assignedCells:
            center = self._constrainedMP.getVertexCoordinates()[0]

            nNodesCells = c.nNodes
            nDof = nNodesCells * self._fieldSize

            N = c.getInterpolationVector(center)

            K = np.outer(N, N) * self._penaltyParameter

            currentValue = N @ dU.reshape((-1, self._fieldSize))

            R_ = PExt[idxP : idxP + nDof]
            K_ = V[idxK : idxK + nDof**2].reshape((nDof, nDof))

            idxP += nDof
            idxK += nDof**2

            for i, prescribedComponent in self._prescribedStepDelta.items():
                R_[i :: self._fieldSize] += (
                    N
                    * self._penaltyParameter
                    * (currentValue[i] - prescribedComponent * timeStep.stepProgressIncrement)
                )
                K_[i :: self._fieldSize, i :: self._fieldSize] += K
