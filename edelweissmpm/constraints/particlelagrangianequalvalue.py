import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmpm.particles.base.baseparticle import BaseParticle


class ParticleLagrangianEqualValueConstraint(MPMConstraintBase):
    """
    This is an implementation of an equal value constraint using Lagrangian multipliers.
    It constrains the field values of two particles to be equal for the specified components.
    The constraint can be expressed as:
    g(x) = value_master - value_slave = 0
    where value_master and value_slave are the interpolated field values at the master and slave particles, respectively.
    They are computed as:
    value_master = N_master * U_master
    value_slave = N_slave * U_slave
    where N_master and N_slave are the interpolation vectors for the master and slave particles,
    and U_master and U_slave are the nodal field values.

    Parameters
    ----------
    name
        The name of this constraint.
    masterParticle
        The master particle whose field value is to be matched.
    components
        The components of the field to be constrained.
    slaveParticle
        The slave particle whose field value is to be constrained to match the master particle.
    field
        The field this constraint is acting on.
    model
        The full MPMModel instance.
    """

    def __init__(
        self,
        name: str,
        masterParticle: BaseParticle,
        components: list[int] | int,
        slaveParticle: BaseParticle,
        field: str,
        model,
    ):
        self._name = name
        self._field = field
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        self._masterParticle = masterParticle
        self._slaveParticle = slaveParticle
        if isinstance(components, int):
            self._components = [components]
        elif isinstance(components, list):
            self._components = components
        else:
            raise Exception("Invalid components format")

        self._components = np.asarray(self._components)
        self._nLagrangianMultipliers = len(self._components)
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
            n: i
            for i, n in enumerate(
                set(kf.node for p in [self._masterParticle, self._slaveParticle] for kf in p.kernelFunctions)
            )
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

        K_UL = K[: -self._nLagrangianMultipliers, -self._nLagrangianMultipliers :]
        K_LU = K[-self._nLagrangianMultipliers :, : -self._nLagrangianMultipliers]

        self.reactionForce.fill(0.0)
        for i, component in enumerate(self._components):

            P_U_i = PExt_U[i :: self._fieldSize]
            dU_U_j = dU_U[i :: self._fieldSize]

            K_UL_j = K_UL[i :: self._fieldSize, :]
            K_LU_j = K_LU[:, i :: self._fieldSize]

            dL_i = dU_L[i]

            N_master = self._masterParticle.getInterpolationVector(self._masterParticle.getCenterCoordinates())
            N_slave = self._slaveParticle.getInterpolationVector(self._slaveParticle.getCenterCoordinates())

            masterNodeIdcs = [self._nodes[kf.node] for kf in self._masterParticle.kernelFunctions]
            slaveNodeIdcs = [self._nodes[kf.node] for kf in self._slaveParticle.kernelFunctions]

            masterValue = N_master @ dU_U_j[masterNodeIdcs]
            slaveValue = N_slave @ dU_U_j[slaveNodeIdcs]

            g_j = masterValue - slaveValue
            dg_i_dU_j_master = N_master
            dg_i_dU_j_slave = -N_slave

            P_U_i[masterNodeIdcs] += dL_i * dg_i_dU_j_master
            P_U_i[slaveNodeIdcs] += dL_i * dg_i_dU_j_slave
            PExt_L[i] += g_j

            K_UL_j[masterNodeIdcs, i] += dg_i_dU_j_master
            K_LU_j[i, masterNodeIdcs] += dg_i_dU_j_master

            K_UL_j[slaveNodeIdcs, i] += dg_i_dU_j_slave
            K_LU_j[i, slaveNodeIdcs] += dg_i_dU_j_slave

            self.reactionForce[i] += dL_i


def ParticleLagrangianEqualValueConstraintOnParticleSetFactory(
    name: str,
    constrainedParticleSet: set[BaseParticle],
    components: list[int] | int,
    field: str,
    model,
) -> list[ParticleLagrangianEqualValueConstraint]:
    """
    Factory function to create a set of ParticleLagrangianEqualValueConstraint instances
    that constrain all particles in the given set to have equal values for the specified field components.

    Parameters
    ----------
    name
        The base name for the constraints.
    constrainedParticleSet
        The set of particles to be constrained.
    components
        The components of the field to be constrained.
    field
        The field this constraint is acting on.
    model
        The full MPMModel instance.

    Returns
    -------
    list[ParticleLagrangianEqualValueConstraint]
        A list of ParticleLagrangianEqualValueConstraint instances.
    """
    if len(constrainedParticleSet) < 2:
        raise ValueError("At least two particles are required to create equal value constraints.")

    constrainedParticleList = list(constrainedParticleSet)
    masterParticle = constrainedParticleList[0]
    constraints = []
    for i, slaveParticle in enumerate(constrainedParticleList[1:]):
        constraintName = f"{name}_eq_{i}"
        constraint = ParticleLagrangianEqualValueConstraint(
            constraintName,
            masterParticle,
            components,
            slaveParticle,
            field,
            model,
        )
        constraints.append(constraint)

    return constraints
