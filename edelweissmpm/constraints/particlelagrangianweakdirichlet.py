import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.particles.base.baseparticle import BaseParticle


class ParticleLagrangianWeakDirichlet(MPMConstraintBase):
    """
    This is an implementation of weak Dirichlet boundary conditions using a penalty formulation.
    It constrains a material point field increment.

    Parameters
    ----------
    name
        The name of this constraint.
    constrainedParticle
        The particle whose field value is to be constrained.
    constrainedLocation
        The location on the particle to apply the constraint. Can be "center" or a vertex
        index (0 to number of vertices - 1).
    field
        The field this constraint is acting on.
    prescribedStepDelta
        A dictionary mapping field component indices to their prescribed step increments.
    model
        The full MPMModel instance.
    """

    def __init__(
        self,
        name: str,
        constrainedParticle: BaseParticle,
        constrainedLocation: int | str,
        field: str,
        prescribedStepDelta: dict,
        model,
    ):
        self._name = name
        self._constrainedParticle = constrainedParticle
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()
        self._constrainedLocation = constrainedLocation

        if isinstance(constrainedLocation, str):
            if constrainedLocation != "center":
                raise ValueError("Constrain must be 'center' or a vertex index.")
        # elif isinstance(constrainedLocation, int):
        #     raise ValueError(f"Constrain must be 'center' or a vertex index, got {type(constrainedLocation)}.")

        self._nLagrangianMultipliers = len(self._prescribedStepDelta)
        self.reactionForce = np.zeros(self._fieldSize)

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
        return len(self._nodes) * self._fieldSize + self._nLagrangianMultipliers

    @property
    def scalarVariables(
        self,
    ) -> list:
        return self._lagrangianMultipliers

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        return self._nLagrangianMultipliers

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        self._lagrangianMultipliers = scalarVariables

    def updateConnectivity(self, model):

        nodes = {n: i for i, n in enumerate(set(kf.node for kf in self._constrainedParticle.kernelFunctions))}

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        return hasChanged

    def applyConstraint(self, dU_: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):

        dU_U = dU_[: -self._nLagrangianMultipliers]
        dU_L = dU_[-self._nLagrangianMultipliers :]
        PExt_U = PExt[: -self._nLagrangianMultipliers]
        PExt_L = PExt[-self._nLagrangianMultipliers :]

        K = V.reshape((self.nDof, self.nDof))

        # K_UU = K[:-self._nLagrangianMultipliers, :-self._nLagrangianMultipliers]
        K_UL = K[: -self._nLagrangianMultipliers, -self._nLagrangianMultipliers :]
        K_LU = K[-self._nLagrangianMultipliers :, : -self._nLagrangianMultipliers]
        # K_LL = K[-self._nLagrangianMultipliers:, -self._nLagrangianMultipliers:]

        self.reactionForce.fill(0.0)
        p = self._constrainedParticle

        if self._constrainedLocation == "center":
            constrainedCoordinates = p.getCenterCoordinates()
        elif isinstance(self._constrainedLocation, int):
            constrainedCoordinates = p.getVertexCoordinates()[self._constrainedLocation]

        for i, prescribedComponent in self._prescribedStepDelta.items():

            P_U_i = PExt_U[i :: self._fieldSize]
            dU_U_j = dU_U[i :: self._fieldSize]

            K_UL_j = K_UL[i :: self._fieldSize, :]
            K_LU_j = K_LU[:, i :: self._fieldSize]

            dL_i = dU_L[i]

            N = p.getInterpolationVector(constrainedCoordinates)

            nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

            mpValue = N @ dU_U_j[nodeIdcs]

            g_i = mpValue - prescribedComponent * timeStep.stepProgressIncrement
            dg_i_dU_j = N

            P_U_i[nodeIdcs] += dL_i * dg_i_dU_j
            PExt_L[i] += g_i

            K_UL_j[nodeIdcs, i] += dg_i_dU_j
            K_LU_j[i, nodeIdcs] += dg_i_dU_j

            self.reactionForce[i] += dL_i


def ParticleLagrangianWeakDirichletOnParticleSetFactory(
    baseName: str,
    particleSet: list[BaseParticle],
    constrainedLocation: int | str,
    field: str,
    prescribedStepDelta: dict,
    model: MPMModel,
):
    constraints = dict()
    for i, p in enumerate(particleSet):
        name = f"{baseName}_{i}"
        constraint = ParticleLagrangianWeakDirichlet(name, p, constrainedLocation, field, prescribedStepDelta, model)
        constraints[name] = constraint

    return constraints
