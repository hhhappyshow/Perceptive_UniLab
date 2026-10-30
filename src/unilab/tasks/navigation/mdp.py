"""Navigation observations, outcomes, and rewards (InstinctMJ mdp port).

Every tensor read is declared through ``tensor_body_names`` /
``tensor_sensor_names`` so the packed read plan serves the terms without
backend-specific access. Progress and terminal rewards are impulses divided
by step_dt before the reward manager applies dt. They are evaluated every
control step, so an immediate reset cannot discard a reward between updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from unilab.envs.mdp.observations import world_to_body_vector
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

ROBOT_COLLISION_GEOMS = (
    "pelvis",
    "left_hip",
    "left_thigh",
    "left_shin",
    "left_linkage_brace",
    "right_hip",
    "right_thigh",
    "right_shin",
    "right_linkage_brace",
    "left_foot1",
    "left_foot2",
    "left_foot3",
    "left_foot4",
    "left_foot5",
    "left_foot6",
    "left_foot7",
    "right_foot1",
    "right_foot2",
    "right_foot3",
    "right_foot4",
    "right_foot5",
    "right_foot6",
    "right_foot7",
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
OBSTACLE_CONTACT_SENSORS = tuple(
    f"obstacle_contact_{index}_{geom}_{field}"
    for index in range(3)
    for geom in ROBOT_COLLISION_GEOMS
    for field in ("found", "force")
)
_REFERENCE_STEP_DT = 0.1  # InstinctMJ's action-difference reward reference interval.


def controller(env: ManagerBasedRlEnv):
    return env.action_manager.get_term("navigation")


def read_obstacle_contacts(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Read this physics sample; the navigation env accumulates it per row."""
    values = env.scene.read_plan.sensor_tensor_views(
        env.scene["robot"], OBSTACLE_CONTACT_SENSORS
    ).values
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for index in range(3):
        for geom in ROBOT_COLLISION_GEOMS:
            found = values[f"obstacle_contact_{index}_{geom}_found"]
            force = values[f"obstacle_contact_{index}_{geom}_force"]
            hit |= (found.reshape(env.num_envs, -1) > 0.5).any(-1)
            hit |= force.reshape(env.num_envs, -1).norm(dim=-1) > 1e-6
    return hit


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
        return OBSTACLE_CONTACT_SENSORS

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
        hit = controller(env).obstacle_contact
        fallen = self._fall(env, gravity_b[:, 2]) & ~hit
        success = (controller(env).distance() < 0.3) & ~hit & ~fallen
        timeout = (env.episode_length_buf >= env.max_episode_length) & ~hit & ~fallen & ~success
        return dict(collision=hit, fall=fallen, success=success, timeout=timeout)[self._kind]


class navigation_reward(ManagerTermBase):
    """progress / success / collision / fall / time / smoothness rewards.

    Every PPO action is executed. Time is a per-second cost; smoothness uses
    action changes normalized to the original 0.1 s interval. Terminal events
    and new closest approaches are paid on the step where they occur.
    """

    entity_name = "robot"

    _KINDS = ("progress", "success", "collision", "fall", "time", "smoothness")

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        kind = cfg.params.get("kind")
        if kind not in self._KINDS:
            raise ValueError(f"navigation_reward kind must be one of {self._KINDS}")
        self._kind = kind
        self._outcome: navigation_outcome | None = None
        if kind in ("success", "collision", "fall"):
            self._outcome = navigation_outcome(
                ManagerTermBaseCfg(func=navigation_outcome, params={"kind": kind}), env
            )

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("robot/pelvis",)

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return OBSTACLE_CONTACT_SENSORS

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        probe = _cold_probe(env, 1)
        if probe is not None:
            return probe
        device = env.device
        term = controller(env)
        if self._kind == "progress":
            # New-closest-approach bookkeeping runs once per step (guarded by
            # the shared reward_step slot); backing away cannot earn it again.
            if term.reward_step != env.common_step_counter:
                distance = term.distance()
                delta = (term.best_distance - distance).clamp_min(0)
                term.best_distance.copy_(torch.minimum(term.best_distance, distance))
                term.progress.copy_(delta)
                term.reward_step = env.common_step_counter
            return term.progress / env.step_dt
        if self._kind == "time":
            return torch.ones(env.num_envs, device=device)
        if self._kind == "smoothness":
            current = env.action_manager.action.clamp(-1, 1)
            previous = env.action_manager.prev_action.clamp(-1, 1)
            # Preserve the integrated cost of a linear action ramp over the
            # same physical interval, independently of the decision frequency.
            return (current - previous).square().sum(-1) * (_REFERENCE_STEP_DT / env.step_dt) ** 2
        assert self._outcome is not None
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
