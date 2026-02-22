"""
Fault-Tolerant Control Allocation (FTCA).

Maps desired body-frame forces and torques to individual engine throttle and
gimbal commands while respecting engine health constraints.  Uses Weighted
Least Squares (WLS) solved via numpy to handle nominal, engine-out, and
gimbal-stuck scenarios with graceful degradation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .engine import EngineCluster, EngineHealth, EngineState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ControlAllocationProblem:
    """Encapsulates one allocation request.

    Parameters
    ----------
    desired_force  : (3,) desired body-frame force  [Fx, Fy, Fz] (N).
    desired_torque : (3,) desired body-frame torque  [Mx, My, Mz] (N*m).
    available_engines : list of engine IDs that may be commanded.
    constraints : dict of any extra solver constraints.
    """
    desired_force: np.ndarray
    desired_torque: np.ndarray
    available_engines: List[int] = field(default_factory=list)
    constraints: Dict = field(default_factory=dict)


@dataclass
class AllocationResult:
    """Result of a control-allocation solve.

    Attributes
    ----------
    throttle_commands : (N,)   throttle per engine [0, 1].
    gimbal_commands   : (N, 2) gimbal [pitch, yaw] per engine (rad).
    achieved_force    : (3,)   force the solution actually produces.
    achieved_torque   : (3,)   torque the solution actually produces.
    residual_norm     : float  ||desired - achieved|| weighted norm.
    is_feasible       : bool   True if residual is near zero.
    """
    throttle_commands: np.ndarray
    gimbal_commands: np.ndarray
    achieved_force: np.ndarray
    achieved_torque: np.ndarray
    residual_norm: float
    is_feasible: bool


# ---------------------------------------------------------------------------
# FTCA Allocator
# ---------------------------------------------------------------------------

class FTCAAllocator:
    """Fault-Tolerant Control Allocation via Weighted Least Squares.

    The allocator builds an *effectiveness matrix* **B** that linearly maps
    the engine command vector **u** to the 6-DOF virtual command
    **v** = [Fx, Fy, Fz, Mx, My, Mz]^T.

    For each available engine *i* the command vector contains three DOFs:
        u_i = [throttle_i, gimbal_pitch_i, gimbal_yaw_i]

    giving  dim(u) = 3 * n_available.

    The allocation problem is:
        min_u || W_v (B u - v_des) ||^2  +  epsilon || W_u u ||^2
    subject to  u_min <= u <= u_max

    Solved iteratively via bounded least-squares (active-set on bounds).
    """

    def __init__(
        self,
        cluster: EngineCluster,
        force_weight: np.ndarray = None,
        torque_weight: np.ndarray = None,
        regularisation: float = 1e-4,
    ):
        self.cluster = cluster
        self.n_engines = cluster.num_engines

        # Weights on virtual-command tracking (6,)
        w_f = force_weight if force_weight is not None else np.array([1.0, 1.0, 1.0])
        w_t = torque_weight if torque_weight is not None else np.array([1.0, 1.0, 1.0])
        self.W_v = np.diag(np.concatenate([w_f, w_t]))  # (6, 6)

        # Regularisation on actuator usage
        self.eps = regularisation

        # Cache for virtual reconfiguration decisions
        self._shutdown_engines: List[int] = []

    # ------------------------------------------------------------------
    # Effectiveness matrix
    # ------------------------------------------------------------------

    def _build_effectiveness_matrix(
        self,
        engine_ids: List[int],
        engine_states: List[EngineState],
        com_offset: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the (6, 3*n) effectiveness matrix B.

        Also returns lower and upper bound vectors for the command vector u.

        For engine *i* located at r_i (relative to CoM) with max thrust F_max:
            Throttle contribution (column 3i):
                Force  = F_max * [sin(yaw_0), -sin(pitch_0)*cos(yaw_0), cos(pitch_0)*cos(yaw_0)]
                Torque = r_i x Force
            Gimbal-pitch perturbation (column 3i+1):
                dForce/d_pitch ≈ F_max * throttle * [-0, -cos(pitch_0)*cos(yaw_0), -sin(pitch_0)*cos(yaw_0)]
                (linearised about current gimbal)
            Gimbal-yaw perturbation (column 3i+2):
                dForce/d_yaw ≈ F_max * throttle * [cos(yaw_0), sin(pitch_0)*sin(yaw_0), -cos(pitch_0)*sin(yaw_0)]
        """
        n = len(engine_ids)
        B = np.zeros((6, 3 * n))
        u_min = np.zeros(3 * n)
        u_max = np.zeros(3 * n)

        state_map = {s.engine_id: s for s in engine_states}

        for idx, eid in enumerate(engine_ids):
            eng = self.cluster.get_engine(eid)
            st = state_map.get(eid)
            if st is None:
                continue

            F_max = eng.max_thrust * eng._efficiency
            r = eng.position - com_offset

            # Current gimbal angles (actual)
            pitch0 = st.gimbal_angles_actual[0]
            yaw0 = st.gimbal_angles_actual[1]
            cp, sp = np.cos(pitch0), np.sin(pitch0)
            cy, sy = np.cos(yaw0), np.sin(yaw0)

            # --- Throttle column (3*idx) ---
            f_per_throttle = F_max * np.array([sy, -sp * cy, cp * cy])
            tau_per_throttle = np.cross(r, f_per_throttle)
            B[:3, 3 * idx] = f_per_throttle
            B[3:, 3 * idx] = tau_per_throttle

            throttle_actual = st.throttle_actual
            if throttle_actual < 1e-6:
                throttle_actual = 0.5  # Linearise about mid-range if engine idle

            # --- Gimbal-pitch column (3*idx + 1) ---
            df_dpitch = F_max * throttle_actual * np.array([0.0, -cp * cy, -sp * cy])
            dtau_dpitch = np.cross(r, df_dpitch)
            B[:3, 3 * idx + 1] = df_dpitch
            B[3:, 3 * idx + 1] = dtau_dpitch

            # --- Gimbal-yaw column (3*idx + 2) ---
            df_dyaw = F_max * throttle_actual * np.array([cy, sp * sy, -cp * sy])
            dtau_dyaw = np.cross(r, df_dyaw)
            B[:3, 3 * idx + 2] = df_dyaw
            B[3:, 3 * idx + 2] = dtau_dyaw

            # Bounds
            is_gimbal_stuck = (st.health == EngineHealth.GIMBAL_STUCK)
            min_thr = eng.min_throttle if st.health != EngineHealth.FAILED_OFF else 0.0
            max_thr = 1.0

            u_min[3 * idx] = 0.0       # throttle can be 0 (off) or >= min_throttle
            u_max[3 * idx] = max_thr

            if is_gimbal_stuck:
                # Fix gimbal DOFs: bounds equal to current value
                u_min[3 * idx + 1] = pitch0
                u_max[3 * idx + 1] = pitch0
                u_min[3 * idx + 2] = yaw0
                u_max[3 * idx + 2] = yaw0
            else:
                u_min[3 * idx + 1] = -eng.max_gimbal
                u_max[3 * idx + 1] = eng.max_gimbal
                u_min[3 * idx + 2] = -eng.max_gimbal
                u_max[3 * idx + 2] = eng.max_gimbal

        return B, u_min, u_max

    # ------------------------------------------------------------------
    # Bounded WLS solver
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_bounded_wls(
        B: np.ndarray,
        v_des: np.ndarray,
        W_v: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        eps: float = 1e-4,
        max_iter: int = 20,
    ) -> np.ndarray:
        """Solve the bounded Weighted Least Squares problem.

        min_u  || W_v (B u - v_des) ||^2  +  eps * ||u||^2
        s.t.   u_min <= u <= u_max

        Uses iterative clipping (projected gradient / active-set lite):
        1. Solve unconstrained WLS.
        2. Clip to bounds.
        3. Fix variables at their bound, remove from optimisation.
        4. Repeat until convergence.
        """
        n = B.shape[1]
        free = np.ones(n, dtype=bool)
        u = 0.5 * (u_min + u_max)  # warm-start at midpoint

        for _iteration in range(max_iter):
            free_idx = np.where(free)[0]
            if len(free_idx) == 0:
                break

            B_f = B[:, free_idx]
            WB = W_v @ B_f
            # Normal equations: (WB^T WB + eps I) u_f = WB^T W_v v_des - contribution of fixed vars
            fixed_idx = np.where(~free)[0]
            v_residual = W_v @ v_des
            if len(fixed_idx) > 0:
                v_residual = v_residual - W_v @ B[:, fixed_idx] @ u[fixed_idx]

            A = WB.T @ WB + eps * np.eye(len(free_idx))
            b = WB.T @ v_residual
            try:
                u_f = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                u_f = np.linalg.lstsq(A, b, rcond=None)[0]

            u[free_idx] = u_f

            # Clip and detect newly-fixed variables
            changed = False
            for j in free_idx:
                if u[j] < u_min[j]:
                    u[j] = u_min[j]
                    free[j] = False
                    changed = True
                elif u[j] > u_max[j]:
                    u[j] = u_max[j]
                    free[j] = False
                    changed = True

            if not changed:
                break

        # Final clip for safety
        u = np.clip(u, u_min, u_max)
        return u

    # ------------------------------------------------------------------
    # Virtual reconfiguration
    # ------------------------------------------------------------------

    def _virtual_reconfigure(
        self,
        engine_ids: List[int],
        engine_states: List[EngineState],
    ) -> List[int]:
        """Determine if any healthy engines should be shut down to maintain
        symmetry and reduce unbalanced torque.

        Strategy: if an engine has FAILED_OFF, shut its diametrically
        opposite engine (if one exists) to neutralise the resulting torque
        bias.  This sacrifices total thrust but simplifies the allocation.
        """
        state_map = {s.engine_id: s for s in engine_states}
        failed_ids = [
            eid for eid in engine_ids
            if state_map[eid].health == EngineHealth.FAILED_OFF
        ]
        if not failed_ids:
            self._shutdown_engines = []
            return engine_ids

        shutdown = set()
        positions = {eid: self.cluster.get_engine(eid).position for eid in engine_ids}

        for fid in failed_ids:
            fpos = positions[fid]
            # Find the engine closest to the diametrically opposite position
            best_eid = None
            best_dist = np.inf
            opposite = -fpos  # Mirror through centre
            for eid in engine_ids:
                if eid == fid or eid in shutdown:
                    continue
                if state_map[eid].health == EngineHealth.FAILED_OFF:
                    continue
                d = np.linalg.norm(positions[eid] - opposite)
                if d < best_dist:
                    best_dist = d
                    best_eid = eid

            # Only shut opposite if it is reasonably close to the mirror point
            if best_eid is not None and best_dist < 0.5:
                shutdown.add(best_eid)

        self._shutdown_engines = list(shutdown)

        # Return engine list excluding both failed and virtually-shutdown
        active = [
            eid for eid in engine_ids
            if state_map[eid].health != EngineHealth.FAILED_OFF
            and eid not in shutdown
        ]
        return active

    # ------------------------------------------------------------------
    # Main allocation entry point
    # ------------------------------------------------------------------

    def allocate(
        self,
        desired_force: np.ndarray,
        desired_torque: np.ndarray,
        engine_states: List[EngineState],
        com_offset: np.ndarray = None,
        enable_virtual_reconfig: bool = True,
    ) -> AllocationResult:
        """Allocate desired forces/torques to engine commands.

        Parameters
        ----------
        desired_force  : (3,) body-frame force  (N).
        desired_torque : (3,) body-frame torque  (N*m).
        engine_states  : list of current EngineState for every engine.
        com_offset     : (3,) current centre-of-mass in body frame.
        enable_virtual_reconfig : if True, may shut healthy engines for
                                  symmetry after engine-out.

        Returns
        -------
        AllocationResult with per-engine throttle and gimbal commands.
        """
        if com_offset is None:
            com_offset = self.cluster._com_offset

        desired_force = np.asarray(desired_force, dtype=np.float64)
        desired_torque = np.asarray(desired_torque, dtype=np.float64)
        v_des = np.concatenate([desired_force, desired_torque])  # (6,)

        # Determine which engines are available
        state_map = {s.engine_id: s for s in engine_states}
        available = [
            eid for eid in range(self.n_engines)
            if state_map[eid].health != EngineHealth.FAILED_OFF
        ]

        # Virtual reconfiguration (optional)
        if enable_virtual_reconfig:
            active = self._virtual_reconfigure(
                list(range(self.n_engines)), engine_states
            )
        else:
            active = available

        if len(active) == 0:
            # No engines -- nothing we can do
            return AllocationResult(
                throttle_commands=np.zeros(self.n_engines),
                gimbal_commands=np.zeros((self.n_engines, 2)),
                achieved_force=np.zeros(3),
                achieved_torque=np.zeros(3),
                residual_norm=np.linalg.norm(v_des),
                is_feasible=False,
            )

        # Build effectiveness matrix for active engines
        B, u_min, u_max = self._build_effectiveness_matrix(
            active, engine_states, com_offset
        )

        # Solve bounded WLS
        u_opt = self._solve_bounded_wls(
            B, v_des, self.W_v, u_min, u_max, self.eps
        )

        # Unpack solution into per-engine commands
        throttle_cmds = np.zeros(self.n_engines)
        gimbal_cmds = np.zeros((self.n_engines, 2))

        for idx, eid in enumerate(active):
            thr = u_opt[3 * idx]
            # Enforce dead-band: if throttle too low, set to 0
            eng = self.cluster.get_engine(eid)
            if thr < eng.min_throttle * 0.5:
                thr = 0.0
            throttle_cmds[eid] = thr
            gimbal_cmds[eid, 0] = u_opt[3 * idx + 1]
            gimbal_cmds[eid, 1] = u_opt[3 * idx + 2]

        # For virtually-shutdown engines, set throttle to 0
        for eid in self._shutdown_engines:
            throttle_cmds[eid] = 0.0
            gimbal_cmds[eid] = 0.0

        # Compute achieved virtual command
        v_achieved = B @ u_opt
        achieved_force = v_achieved[:3]
        achieved_torque = v_achieved[3:]
        residual = np.linalg.norm(self.W_v @ (v_achieved - v_des))
        is_feasible = residual < 1e-2 * (np.linalg.norm(v_des) + 1.0)

        return AllocationResult(
            throttle_commands=throttle_cmds,
            gimbal_commands=gimbal_cmds,
            achieved_force=achieved_force,
            achieved_torque=achieved_torque,
            residual_norm=float(residual),
            is_feasible=is_feasible,
        )

    # ------------------------------------------------------------------
    # Convenience: build and solve a ControlAllocationProblem
    # ------------------------------------------------------------------

    def solve_problem(
        self,
        problem: ControlAllocationProblem,
        engine_states: List[EngineState],
        com_offset: np.ndarray = None,
    ) -> AllocationResult:
        """Solve a pre-packaged ControlAllocationProblem."""
        return self.allocate(
            desired_force=problem.desired_force,
            desired_torque=problem.desired_torque,
            engine_states=engine_states,
            com_offset=com_offset,
        )
