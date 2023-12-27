from fe.config.phenomena import getFieldSize
from fe.variables.scalarvariable import ScalarVariable
from fe.timesteppers.timestep import TimeStep
from mpm.models.mpmmodel import MPMModel
from mpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from mpm.materialpoints.base.mp import MaterialPointBase
import numpy as np


class PenaltyEqualValue(MPMConstraintBase):
    """
    This is an implementation of an equal value constraint using a penalty formulation.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The full MPMModel instance.
    constrainedMaterialPoints
        The list of material points to be constrained.
    field
        The field this constraint is acting on.
    prescribedComponent
        The index of the constrained component.
    penaltyParameter
        The penalty parameter value.
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        constrainedMaterialPoints: list[MaterialPointBase],
        field: str,
        prescribedComponent: int,
        penaltyParameter: float,
    ):
        self._name = name
        self._model = model
        self._constrainedMPs = constrainedMaterialPoints
        self._field = field
        self._prescribedComponent = prescribedComponent
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
        self._nodes = [n for mp in self._constrainedMPs for c in mp.assignedCells for n in c.nodes]

    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):
        P_Ai = PExt.reshape((-1, self._fieldSize))
        dU_Bj = dU.reshape((-1, self._fieldSize))

        K_AiBj = V.reshape(P_Ai.shape + dU_Bj.shape)

        i = self._prescribedComponent

        # Part 1: compute the mean values over all material points

        dMeanValue_dDU_Bj = np.zeros_like(dU_Bj)
        currentNodeIdx = 0
        meanValue = 0.0
        for mp in self._constrainedMPs:
            center = mp.getCenterCoordinates()

            for c in mp.assignedCells:
                nNodesCell = c.nNodes
                nodeIdcs = slice(currentNodeIdx, currentNodeIdx + nNodesCell)

                N = c.getInterpolationVector(center)

                meanValue += N @ dU_Bj[nodeIdcs, i]
                dMeanValue_dDU_Bj[nodeIdcs, i] += N

                currentNodeIdx += nNodesCell

        meanValue /= len(self._constrainedMPs)
        dMeanValue_dDU_Bj /= len(self._constrainedMPs)

        # Part 2: compute the penalized difference between mean value and mp value:
        currentNodeIdx = 0
        for mp in self._constrainedMPs:
            center = mp.getCenterCoordinates()

            for c in mp.assignedCells:
                nNodesCell = c.nNodes

                N = c.getInterpolationVector(center)

                nodeIdcs = slice(currentNodeIdx, currentNodeIdx + nNodesCell)

                mpValue = N @ dU_Bj[nodeIdcs, i]

                P_Ai[nodeIdcs, i] += N * self._penaltyParameter * (mpValue - meanValue)

                K_AiBj[nodeIdcs, i, nodeIdcs, i] += np.outer(N, N) * self._penaltyParameter
                K_AiBj[nodeIdcs, i, :, i] += np.outer(N, dMeanValue_dDU_Bj[:, i]) * -1 * self._penaltyParameter

                currentNodeIdx += nNodesCell
