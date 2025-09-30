import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.particles.base.baseparticle import BaseParticle


class ParticlePenaltyWeakDirichlet(MPMConstraintBase):
    """
    This is an implementation of weak Dirichlet boundary conditions using a penalty formulation.
    It constrains a material point field increment.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The full MPMModel instance.
    constrainedParticles
        The list of particles to be constrained.
    field
        The field this constraint is acting on.
    prescribedStepDelta
        The dictionary containing the prescribed bc components for the field in the present load step.
    penaltyParameter
        The penalty parameter value.
    constrain
        Either constrain the center of the particle or a list of vertex indices for particles with multiple vertices.
        If "center", the center of the particle is constrained. If a list, the vertices at the given indices are constrained.
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        constrainedParticles: list[BaseParticle],
        field: str,
        prescribedStepDelta: dict,
        penaltyParameter: float,
        constrain: str | list[int] = "center",
        **kwargs,
    ):
        self._name = name
        self._model = model
        self._constrainedParticles = constrainedParticles
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._penaltyParameter = penaltyParameter
        self._nodes = dict()
        if constrain == "center":
            self._constrainVertices = None
        else:
            if not isinstance(constrain, list):
                raise ValueError("Constrain must be 'center' or a list of vertex indices.")
            if len(constrain) > 0 and not all(isinstance(i, int) for i in constrain):
                raise ValueError("Constrain must be 'center' or a list of vertex indices.")
            self._constrainVertices = constrain

        self.penaltyForce = np.zeros(self._fieldSize)

        if "f_t" in kwargs:
            self._f_t = kwargs["f_t"]
        else:
            self._f_t = lambda x: x

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
            n: i for i, n in enumerate(set(kf.node for p in self._constrainedParticles for kf in p.kernelFunctions))
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

            for p in self._constrainedParticles:
                if self._constrainVertices:
                    constrainedCoordinates = p.getVertexCoordinates()[self._constrainVertices]
                else:
                    constrainedCoordinates = [p.getCenterCoordinates()]

                for constrainedCoordinate in constrainedCoordinates:

                    N = p.getInterpolationVector(constrainedCoordinate)

                    nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

                    mpValue = N @ dU_j[nodeIdcs]

                    P_i[nodeIdcs] += (
                        N
                        * self._penaltyParameter
                        * (
                            mpValue
                            - prescribedComponent
                            * (
                                self._f_t(timeStep.stepProgress)
                                - self._f_t(timeStep.stepProgress - timeStep.stepProgressIncrement)
                            )
                        )
                    )
                    K_ij[np.ix_(nodeIdcs, nodeIdcs)] += np.outer(N, N) * self._penaltyParameter

            self.penaltyForce[i] = np.sum(P_i)
