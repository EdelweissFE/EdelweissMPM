import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmpm.particles.base.baseparticle import BaseParticle


class ParticleLagrangianWeakDirichlet(MPMConstraintBase):
    """
    This is an implementation of weak Dirichlet boundary conditions using a penalty formulation.
    It constrains a material point field increment.

    Parameters
    ----------
    name
        The name of this constraint.
    constrainedParticles
        The dictionary of particles and their constraints.
        Currently, the only supported constraints are:
        - "center": Constrain the center of the particle.
        - list of vertex indices: Constrain the particle at the specified vertex indices.
    field
        The field this constraint is acting on.
    prescribedStepDelta
        The dictionary containing the prescribed bc components for the field in the present load step.
    model
        The full MPMModel instance.
    """

    def __init__(
        self,
        name: str,
        constrainedParticles: dict[BaseParticle, str | list],
        field: str,
        prescribedStepDelta: dict,
        model,
    ):
        self._name = name
        self._constrainedParticles = constrainedParticles
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        self._constrainedLocations = []

        for p, constrain in self._constrainedParticles.items():
            if isinstance(constrain, str):
                if constrain != "center":
                    raise ValueError("Constrain must be 'center' or a list of vertex indices.")
                self._constrainedLocations.append((p, "center"))
            elif isinstance(constrain, list):
                if len(constrain) > 0 and not all(isinstance(i, int) for i in constrain):
                    raise ValueError("Constrain must be 'center' or a list of vertex indices.")
                for i in constrain:
                    self._constrainedLocations.append((p, i))
            else:
                raise ValueError("Constrain must be 'center' or a list of vertex indices.")

        self._nLagrangianMultipliers = len(self._constrainedLocations) * len(self._prescribedStepDelta)
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
        nodes = {
            n: i for i, n in enumerate(set(kf.node for p in self._constrainedParticles for kf in p.kernelFunctions))
        }

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

        currentConstraint = 0
        self.reactionForce.fill(0.0)
        for i, prescribedComponent in self._prescribedStepDelta.items():

            P_U_i = PExt_U[i :: self._fieldSize]
            dU_U_j = dU_U[i :: self._fieldSize]

            K_UL_j = K_UL[i :: self._fieldSize, :]
            K_LU_j = K_LU[:, i :: self._fieldSize]

            for p, constraintLocation in self._constrainedLocations:

                if constraintLocation == "center":
                    constrainedCoordinates = p.getCenterCoordinates()
                elif isinstance(constraintLocation, int):
                    constrainedCoordinates = p.getVertexCoordinates()[constraintLocation]

                c = currentConstraint

                dL_c = dU_L[c]

                N = p.getInterpolationVector(constrainedCoordinates)

                nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

                mpValue = N @ dU_U_j[nodeIdcs]

                g_j = mpValue - prescribedComponent * timeStep.stepProgressIncrement
                dg_i_dU_j = N

                P_U_i[nodeIdcs] += dL_c * dg_i_dU_j
                PExt_L[c] += g_j

                K_UL_j[nodeIdcs, c] += dg_i_dU_j
                K_LU_j[c, nodeIdcs] += dg_i_dU_j

                self.reactionForce[i] += dL_c

                currentConstraint += 1
