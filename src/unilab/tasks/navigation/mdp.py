"""Navigation observations, outcomes, and rewards (InstinctMJ mdp port).

Every tensor read is declared through ``tensor_body_names`` /
``tensor_sensor_names`` so the packed read plan serves the terms without
backend-specific access. Reward values reproduce InstinctMJ's per-0.1 s
quantities on high-level boundary steps and emit zero elsewhere; with
``scale_rewards_by_dt=True`` and a 0.02 s step this keeps per-interval reward
totals identical to the 10 Hz training contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from unilab.envs.mdp.observations import world_to_body_vector
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

_TORSO_SUBTREE_GEOMS = (
    "torso",
    "head",
    "left_shoulder_yaw",
    "left_elbow_yaw",
    "left_wrist",
    "left_hand",
    "right_shoulder_yaw",
    "right_elbow_yaw",
    "right_wrist",
    "right_hand",
)
_OBSTACLE_CONTACT_SENSORS = tuple(
    f"obstacle_contact_{index}_{geom}_{field}"
    for index in range(3)
    for geom in _TORSO_SUBTREE_GEOMS
    for field in ("found", "force")
)
_BOUNDARY_VALUE_SCALE = 5.0  # 0.1 s trained interval / 0.02 s env step


def controller(env: ManagerBasedRlEnv):
    return env.action_manager.get_term("navigation")


def _cold_probe(env: ManagerBasedRlEnv, width: int, *, dtype: torch.dtype = torch.float32):
    """Shape probe before the packed read plan exists (ObservationManager cold
    path probes term widths by calling them once)."""
    if getattr(env.scene, "read_plan", None) is None:
        return torch.zeros((env.num_envs, width), dtype=dtype, device=env.device)
    return None


def _pelvis_view(env: ManagerBasedRlEnv, term_name: str):
    read_plan = getattr(env.scene, "read_plan", None)
    if read_plan is None:
        raise NotImplementedError(f"{term_name} requires the packed pelvis read plan")
    return read_plan.body_tensor_view(env.scene["robot"], ("robot/pelvis",))


class goal(ManagerTermBase):
    """Body-frame goal observation; the teleop pseudo-goal while teleoping."""

    entity_name = "robot"

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("robot/pelvis",)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        probe = _cold_probe(env, 3)
        if probe is not None:
            return probe
        return controller(env).relative_goal()


class navigation_outcome(ManagerTermBase):
    """success / collision / fall / timeout outcomes in one packed read."""

    entity_name = "robot"

    _KINDS = ("success", "collision", "fall", "timeout")

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        kind = cfg.params.get("kind")
        if kind not in self._KINDS:
            raise ValueError(f"navigation_outcome kind must be one of {self._KINDS}")
        self._kind = kind

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("robot/pelvis",)

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return _OBSTACLE_CONTACT_SENSORS

    def _collision(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        read_plan = env.scene.read_plan
        values = read_plan.sensor_tensor_views(
            env.scene["robot"], _OBSTACLE_CONTACT_SENSORS
        ).values
        hits = []
        for index in range(3):
            found_any = None
            force_any = None
            for geom in _TORSO_SUBTREE_GEOMS:
                found = values[f"obstacle_contact_{index}_{geom}_found"] > 0.5
                force = values[f"obstacle_contact_{index}_{geom}_force"]
                recent = force.reshape(env.num_envs, -1).norm(dim=-1) > 1e-6
                found_any = found if found_any is None else (found_any | found)
                force_any = recent if force_any is None else (force_any | recent)
            hits.append(found_any.reshape(env.num_envs, -1).any(-1) | force_any)
        return torch.stack(hits).any(0)

    def _fall(self, env: ManagerBasedRlEnv, gravity_b_z: torch.Tensor) -> torch.Tensor:
        view = _pelvis_view(env, "navigation_outcome")
        origin = env.scene.env_origins(env.device)
        height = view.pos_w[:, 0, 2] - origin[:, 2]
        return (gravity_b_z > -0.5) | (height < 0.45)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        probe = _cold_probe(env, 1, dtype=torch.bool)
        if probe is not None:
            return probe
        view = _pelvis_view(env, "navigation_outcome")
        quat_w = view.quat_w[:, 0, :]
        gravity = env.scene["robot"].data.gravity_vec_w(env.device)
        if gravity.ndim == 1:
            gravity = gravity[None, :]
        gravity_b = world_to_body_vector(quat_w, gravity.expand(quat_w.shape[0], 3))
        hit = self._collision(env)
        fallen = self._fall(env, gravity_b[:, 2]) & ~hit
        success = (controller(env).distance() < 0.3) & ~hit & ~fallen
        timeout = (env.episode_length_buf >= env.max_episode_length) & ~hit & ~fallen & ~success
        return dict(
            collision=hit, fall=fallen, success=success, timeout=timeout
        )[self._kind]


class navigation_reward(ManagerTermBase):
    """progress / success / collision / fall / time / smoothness rewards.

    Values are InstinctMJ's per-0.1 s quantities scaled onto high-level
    boundary steps (every ``high_level_every`` 0.02 s steps) and zero between
    boundaries, so per-interval totals match the 10 Hz training contract.
    """

    entity_name = "robot"

    _KINDS = ("progress", "success", "collision", "fall", "time", "smoothness")

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        kind = cfg.params.get("kind")
        if kind not in self._KINDS:
            raise ValueError(f"navigation_reward kind must be one of {self._KINDS}")
        self._kind = kind
        self._outcome = None
        if kind in ("success", "collision", "fall"):
            self._outcome = navigation_outcome(
                ManagerTermBaseCfg(func=navigation_outcome, params={"kind": kind}), env
            )

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("robot/pelvis",)

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return _OBSTACLE_CONTACT_SENSORS

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        probe = _cold_probe(env, 1)
        if probe is not None:
            return probe
        device = env.device
        term = controller(env)
        boundary = term.high_level_boundary
        if self._kind == "progress":
            # New-closest-approach bookkeeping runs once per step (guarded by
            # the shared reward_step slot); backing away cannot earn it again.
            if term.reward_step != env.common_step_counter:
                distance = term.distance()
                delta = (term.best_distance - distance).clamp_min(0)
                term.best_distance.copy_(torch.minimum(term.best_distance, distance))
                term.progress.copy_(delta)
                term.interval_progress += delta
                term.reward_step = env.common_step_counter
            if not boundary:
                return torch.zeros(env.num_envs, device=device)
            value = term.interval_progress / env.step_dt
            term.interval_progress.zero_()
            return value
        if not boundary:
            return torch.zeros(env.num_envs, device=device)
        if self._kind == "time":
            # InstinctMJ charged -0.1 weight per 0.1 s interval (value 1).
            return torch.full((env.num_envs,), _BOUNDARY_VALUE_SCALE, device=device)
        if self._kind == "smoothness":
            current = env.action_manager.action.clamp(-1, 1)
            previous = env.action_manager.prev_action.clamp(-1, 1)
            return (current - previous).square().sum(-1) * _BOUNDARY_VALUE_SCALE
        return self._outcome(env).float() / env.step_dt


def reset_robot_to_stand(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Composed-scene reset: robot root to the env origin, joints to 'stand'.

    The mapped (composed) entity scene supports the tensor root write but has
    no entity-patch joint write on the device-scoped reset path, so the root
    goes through the public entity API and the default joint columns go
    through the packed reset transaction by their qpos/qvel indices.
    """
    robot = env.scene["robot"]
    device = env.device
    reset_mask = env.event_term_mask_buffer.scatter(env_ids)
    default_root = robot.data.default_root_state(device)
    origins = env.scene.env_origins(device)
    root_state = default_root.clone()
    root_state[:, :3] += origins
    robot.write_root_state_tensor_to_sim(root_state, reset_mask)
    joint_pos = robot.data.default_joint_pos(device)
    qpos_ids = np.asarray(env.backend.get_joint_state_qpos_indices(tuple(robot.joint_names)))
    qvel_ids = np.asarray(env.backend.get_joint_state_qvel_indices(tuple(robot.joint_names)))
    env.scene._reset_state.write_joint_state_mask(
        reset_mask,
        qpos_ids,
        qvel_ids,
        joint_pos,
        torch.zeros_like(joint_pos),
        term_name="navigation_reset_robot_to_stand",
    )
    # Teleport the obstacle boxes and plan the goal inside the transaction.
    controller(env).place_obstacles(env_ids)


class base_ang_vel_body(ManagerTermBase):
    """Root-body angular velocity in the body frame (root_link_ang_vel_b)."""

    entity_name = "robot"

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("robot/pelvis",)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        probe = _cold_probe(env, 3)
        if probe is not None:
            return probe
        view = _pelvis_view(env, "base_ang_vel_body")
        return world_to_body_vector(view.quat_w[:, 0, :], view.ang_vel_w[:, 0, :])


class projected_gravity_body(ManagerTermBase):
    """Root-body projected gravity (R^T g-hat) from the packed body pose."""

    entity_name = "robot"

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("robot/pelvis",)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        probe = _cold_probe(env, 3)
        if probe is not None:
            return probe
        view = _pelvis_view(env, "projected_gravity_body")
        gravity = env.scene["robot"].data.gravity_vec_w(env.device)
        if gravity.ndim == 1:
            gravity = gravity[None, :]
        # R^T g: upright -> (0,0,-1), the mjlab/InstinctMJ convention.
        return world_to_body_vector(view.quat_w[:, 0, :], gravity.expand(view.quat_w.shape[0], 3))