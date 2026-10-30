"""Two-rate residual navigation action for UniLab (InstinctMJ port).

The high-level residual policy decides a velocity command at 10 Hz; the frozen
official unitree ONNX locomotion policy tracks it at 50 Hz. UniLab applies one
control per env step, so the port runs the env at 50 Hz (sim_dt 0.005,
ctrl_dt 0.02) and the action term consumes the high-level action only every
``high_level_every`` steps (default 5 -> 0.1 s), matching InstinctMJ's two-rate
decimation exactly.

Keyboard teleop (Instinct ``--teleop`` semantics adapted to UniLab commander
channels): arrows nudge ``[vx, vy, vyaw]`` increments, Enter zeroes the
command. While a teleop command is active the trained residual is discarded
("the residual must not walk on its own") and a hard-coded rule layer stops
forward motion within 0.8 m of a box and sidesteps toward the clearer side.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.envs.manager_based_rl_env import (
    ManagerBasedRlEnv as _ConcreteManagerBasedRlEnv,
)
from unilab.envs.manager_based_rl_env import (
    ManagerBasedRlEnvCfg,
    _resolve_backend_entity_contract,
)
from unilab.envs.mdp.actions.actions import JointPositionActionCfg
from unilab.envs.mdp.observations import world_to_body_vector
from unilab.managers.action_manager import ActionTerm, ActionTermCfg
from unilab.managers.manager_base import ManagerTermBaseCfg
from unilab.tasks.navigation.low_level import (
    OFFICIAL_JOINT_ORDER,
    UnitreeOnnxLocomotion,
)
from unilab.tasks.navigation.ray_observation import (
    NormalizedRayScan,
    clearer_side_speed,
    forward_blocked,
    make_forward_scan_pattern,
)

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

# GLFW keycodes used by the MuJoCo viewer key callback (mirrors
# unilab.scripts.play_interactive).
_GLF_KEY_ENTER = 257
_GLF_KEY_RIGHT, _GLF_KEY_LEFT, _GLF_KEY_DOWN, _GLF_KEY_UP = 262, 263, 264, 265

_HIGH_LEVEL_PERIOD = 0.1  # seconds; the cadence the residual policy trained on


@dataclass(kw_only=True)
class NavigationActionCfg(ActionTermCfg):
    """Configuration for the two-rate residual navigation action term."""

    stage: int = 1
    checkpoint: str = ""
    backend: str = "unitree_onnx"
    teleop: bool = False
    log_goal: bool = False
    show_rays: bool = False
    high_level_every: int = 5

    def build(self, env: ManagerBasedRlEnv) -> "NavigationAction":
        return NavigationAction(self, env)


class NavigationAction(ActionTerm):
    """High-level residual (3-dim) + frozen official low-level gait (29-dim)."""

    uses_tensor_state_packet = True

    def __init__(self, cfg: NavigationActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._device = torch.device(env.device)
        if cfg.backend != "unitree_onnx":
            raise ValueError(
                "UniLab navigation port supports backend='unitree_onnx' only; "
                f"got {cfg.backend!r}"
            )
        self.policy = UnitreeOnnxLocomotion(cfg.checkpoint or None, device=str(self._device))
        entity = self._entity
        local_joint_names = tuple(name.split("/", 1)[-1] for name in entity.joint_names)
        if local_joint_names != OFFICIAL_JOINT_ORDER:
            raise ValueError("Official ONNX requires the pelvis-base G1 joint order")

        # InstinctMJ's validated low-level scaling: the flat task __post_init__
        # overrides the nominal scale with beyondmimic_action_scale =
        # 0.25 * effort_limit / stiffness per actuator group (= deploy.yaml's
        # per-joint scales), NOT the uniform 0.5 declared in the dataclass.
        scale = {
            ".*_hip_pitch_joint": 0.25 * 88.0 / 40.179,
            ".*_hip_yaw_joint": 0.25 * 88.0 / 40.179,
            "robot/waist_yaw_joint": 0.25 * 88.0 / 40.179,
            ".*_hip_roll_joint": 0.25 * 139.0 / 99.098,
            ".*_knee_joint": 0.25 * 139.0 / 99.098,
            ".*_ankle_pitch_joint": 0.25 * 50.0 / 28.501,
            ".*_ankle_roll_joint": 0.25 * 50.0 / 28.501,
            "robot/waist_roll_joint": 0.25 * 50.0 / 28.501,
            "robot/waist_pitch_joint": 0.25 * 50.0 / 28.501,
            ".*_shoulder_pitch_joint": 0.25 * 25.0 / 14.251,
            ".*_shoulder_roll_joint": 0.25 * 25.0 / 14.251,
            ".*_shoulder_yaw_joint": 0.25 * 25.0 / 14.251,
            ".*_elbow_joint": 0.25 * 25.0 / 14.251,
            ".*_wrist_roll_joint": 0.25 * 25.0 / 14.251,
            ".*_wrist_pitch_joint": 0.25 * 5.0 / 16.778,
            ".*_wrist_yaw_joint": 0.25 * 5.0 / 16.778,
        }
        low_cfg = JointPositionActionCfg(
            entity_name=cfg.entity_name,
            actuator_names=(".*",),
            scale=scale,
            use_default_offset=True,
        )
        self.low = low_cfg.build(env)
        if self.low.action_dim != 29:
            raise ValueError("Expected 29 native G1 joint actions")
        if tuple(name.split("/", 1)[-1] for name in self.low.target_names) != OFFICIAL_JOINT_ORDER:
            raise ValueError("Low-level joint targets are not in official ONNX order")

        if not isinstance(cfg.high_level_every, int) or cfg.high_level_every < 1:
            raise ValueError("high_level_every must be a positive integer")
        high_level_dt = env.step_dt * cfg.high_level_every
        if abs(high_level_dt - _HIGH_LEVEL_PERIOD) > 1e-6:
            raise ValueError(
                "high_level_every * ctrl_dt must equal the 0.1 s residual cadence; "
                f"got {high_level_dt} s"
            )
        if abs(self.policy.control_dt - env.step_dt) > 1e-9:
            raise ValueError(
                "The official low-level policy requires a 0.02 s control step; "
                f"env ctrl_dt is {env.step_dt}"
            )
        self.high_level_every = cfg.high_level_every
        self.high_level_boundary = False
        self.stage = cfg.stage
        self._high_level_counter = 0
        self.reward_step = -1

        num_envs = env.num_envs
        self._raw = torch.zeros(num_envs, 3, device=self._device)
        self.velocity = torch.zeros_like(self._raw)
        self.goal = torch.zeros_like(self._raw)
        self.previous_low = torch.zeros(num_envs, 29, device=self._device)
        self.low_time = torch.zeros(num_envs, device=self._device)
        self.best_distance = torch.zeros(num_envs, device=self._device)
        self.progress = torch.zeros(num_envs, device=self._device)
        self.interval_progress = torch.zeros(num_envs, device=self._device)
        # Goal direction/distance planned by place_obstacles (reset event).
        self._planned_angle = torch.zeros(num_envs, device=self._device)
        self._planned_distance = torch.zeros(num_envs, device=self._device)
        self.forward_blocked = torch.zeros(num_envs, dtype=torch.bool, device=self._device)

        self._scan = NormalizedRayScan(
            ManagerTermBaseCfg(func=NormalizedRayScan, params={"ray_pattern": make_forward_scan_pattern()}),
            env,
        )

        self._teleop_lock = threading.Lock()
        self._teleop: list[float] | None = None
        if cfg.teleop:
            self.enable_teleop()

        self._mocap_entities = tuple(
            (env.scene[f"obstacle_{index}"], f"obstacle_{index}/obstacle_box_{index}")
            for index in range(3)
        )
        self._mocap_bound = False

    def bind_mocap_bodies(self) -> None:
        """Bind the three obstacle mocap bodies on the cold path.

        The backend entity scene materializes after the action manager is
        built, so the env subclass calls this once at the end of env init.
        """
        if self._mocap_bound:
            return
        for obstacle, body_name in self._mocap_entities:
            obstacle.bind_mocap_pose_write(body_name, term_name="navigation")
        self._mocap_bound = True

    def playback_debug_overlay_getter(self):
        """Per-frame playback overlay: goal marker and forward rays.

        Ports InstinctMJ's ``NavigationAction.debug_vis`` (goal sphere) and the
        mjlab scanner ``show_rays`` visualization: only hit rays render, as a
        short green line to their hit point plus a small endpoint dot; miss
        rays draw nothing. Primitives are
        environment local (the renderer applies grid offsets).
        """
        if not self.cfg.log_goal:
            return None
        from dataclasses import dataclass

        import numpy as np
        from unisim.backend.base import DebugPrimitive
        from unilab.visualization.debug_primitives import quat_from_z_axis

        @dataclass
        class _LineSegment:
            """Duck-typed line primitive (no arrowhead); native-viewer only."""

            kind: str = "line"
            pos: tuple = (0.0, 0.0, 0.0)
            quat: tuple | None = None
            size: tuple = ()
            rgba: tuple = (0.5, 0.7, 1.0, 0.18)

        def _line(start, end, rgba):
            delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
            length = float(np.linalg.norm(delta))
            if length < 1e-6:
                return None
            return _LineSegment(
                pos=tuple(float(v) for v in start),
                quat=quat_from_z_axis(delta),
                size=(length,),
                rgba=tuple(float(v) for v in rgba),
            )

        device = self._device
        num_envs = self.num_envs
        show_rays = bool(self.cfg.show_rays)
        ray_stride = 2  # hit lines for 256 of 512 rays keeps the viewer under maxgeom
        # mjlab RayCastSensor VizCfg colors: hit rays green, endpoint dots cyan.
        hit_rgba = (0.0, 1.0, 0.0, 0.8)
        dot_rgba = (0.0, 0.9, 1.0, 0.6)
        dot_radius = 0.03
        max_distance = self._scan.ray_pattern.max_distance

        def _get_overlay():
            self._scan.ensure_bound(self._env)
            per_env: list[list] = [[] for _ in range(num_envs)]
            origins_env = self._env.scene.env_origins(device).detach().cpu().numpy()
            goals = self.goal.detach().cpu().numpy()
            for index in range(num_envs):
                gx, gy, _ = goals[index]
                per_env[index].append(
                    DebugPrimitive(
                        kind="sphere",
                        pos=(
                            float(gx - origins_env[index, 0]),
                            float(gy - origins_env[index, 1]),
                            0.25,
                        ),
                        size=(0.08,),
                        rgba=(1.0, 0.15, 0.1, 0.9),
                    )
                )
            if show_rays and getattr(self._env.scene, "read_plan", None) is not None and self._env.scene.read_plan.ready:
                ray_origins, ray_endpoints = self._scan.ray_debug(self._env)
                for index in range(num_envs):
                    ox, oy = origins_env[index, 0], origins_env[index, 1]
                    origins = ray_origins[index]
                    endpoints = ray_endpoints[index]
                    for ray in range(origins.shape[0]):
                        start = origins[ray]
                        end = endpoints[ray]
                        hit = bool(np.linalg.norm(end - start) < max_distance - 1e-6)
                        if not hit:
                            continue
                        per_env[index].append(
                            DebugPrimitive(
                                kind="sphere",
                                pos=(
                                    float(end[0] - ox),
                                    float(end[1] - oy),
                                    float(end[2]),
                                ),
                                size=(dot_radius,),
                                rgba=dot_rgba,
                            )
                        )
                        if ray % ray_stride == 0:
                            primitive = _line(
                                (float(start[0] - ox), float(start[1] - oy), float(start[2])),
                                (float(end[0] - ox), float(end[1] - oy), float(end[2])),
                                hit_rgba,
                            )
                            if primitive is not None:
                                per_env[index].append(primitive)
            return per_env

        return _get_overlay

    # -- manager plumbing ---------------------------------------------------

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("robot/pelvis",)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    @property
    def action_dim(self) -> int:
        return 3

    # -- teleop -------------------------------------------------------------

    def enable_teleop(self) -> None:
        """Arrow-key command replaces the fixed goal. The residual is discarded."""
        with self._teleop_lock:
            self._teleop = [0.0, 0.0, 0.0]

    def clear_teleop(self) -> None:
        with self._teleop_lock:
            self._teleop = [0.0, 0.0, 0.0]
        print("teleop vx=+0.0 vy=+0.0 vyaw=+0.0")

    def nudge_teleop(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> None:
        limits = self.policy.limits.tolist()
        with self._teleop_lock:
            if self._teleop is None:
                self._teleop = [0.0, 0.0, 0.0]
            updated = [
                min(max(self._teleop[0] + vx, limits[0][0]), limits[0][1]),
                min(max(self._teleop[1] + vy, limits[1][0]), limits[1][1]),
                min(max(self._teleop[2] + vyaw, limits[2][0]), limits[2][1]),
            ]
            self._teleop = updated
        print(f"teleop vx={updated[0]:+.1f} vy={updated[1]:+.1f} vyaw={updated[2]:+.1f}")

    def _teleop_state(self) -> tuple[float, float, float] | None:
        with self._teleop_lock:
            if self._teleop is None:
                return None
            return tuple(self._teleop)

    @property
    def teleop_active(self) -> bool:
        return self._teleop is not None

    # -- goal ---------------------------------------------------------------

    def _pelvis_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        view = self._env.scene.read_plan.body_tensor_view(self._entity, ("robot/pelvis",))
        return view.pos_w[:, 0, :], view.quat_w[:, 0, :]

    def relative_goal(self) -> torch.Tensor:
        """Body-frame goal [local_x, local_y, yaw], or the teleop pseudo-goal."""
        command = self._teleop_state()
        if command is not None:
            vx, vy, _vyaw = command
            if abs(vx) < 0.05 and abs(vy) < 0.05:
                return torch.zeros(self.num_envs, 3, device=self._device)
            # Four meters matches the goal distance the residual policy was
            # trained on.
            bearing = math.atan2(vy, vx)
            goal = torch.tensor(
                [[4.0 * math.cos(bearing), 4.0 * math.sin(bearing), 0.0]],
                device=self._device,
            ).expand(self.num_envs, 3)
            return goal
        pos_w, quat_w = self._pelvis_pose()
        delta = self.goal - pos_w
        delta[:, 2] = 0
        local = world_to_body_vector(quat_w, delta)
        return torch.cat((local[:, :2], torch.atan2(local[:, 1], local[:, 0])[:, None]), -1)

    def distance(self) -> torch.Tensor:
        pos_w, _ = self._pelvis_pose()
        return (self.goal[:, :2] - pos_w[:, :2]).norm(dim=-1)

    # -- action lifecycle ----------------------------------------------------

    def _high_level_command(self, actions: torch.Tensor) -> torch.Tensor:
        """Velocity command for one 0.1 s interval, from goal or teleop."""
        command = self._teleop_state()
        if command is None:
            goal = self.relative_goal()
            nominal = torch.stack((0.5 * goal[:, 0], 0.5 * goal[:, 1], goal[:, 2]), -1)
            nominal = self.policy.command(nominal, torch.zeros_like(nominal))
            # Symmetric residual capacity is derived from the saved command intervals.
            scale = (self.policy.limits[:, 1] - self.policy.limits[:, 0]) / 2
            return self.policy.command(nominal, actions.clamp(-1, 1) * scale)
        nominal = torch.tensor([command], device=self._device).expand(self.num_envs, 3)
        # Arrow keys are the velocity command. The trained residual must not
        # walk on its own.
        return self.policy.command(nominal, torch.zeros_like(nominal))

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw.copy_(actions)
        boundary = self._high_level_counter % self.high_level_every == 0
        self.high_level_boundary = boundary
        self._high_level_counter += 1
        if boundary:
            self.velocity.copy_(self._high_level_command(actions))
        command = self._teleop_state()
        if command is not None:
            # The rule layer runs every 50 Hz step while teleoping.
            self._scan.ensure_bound(self._env)
            self.forward_blocked = forward_blocked(self._scan, self._env)
            blocked = self.forward_blocked & (self.velocity[:, 0] > 0.05)
            self.velocity[:, 0] = torch.where(
                blocked, torch.zeros_like(self.velocity[:, 0]), self.velocity[:, 0]
            )
            self.velocity[:, 1] = torch.where(
                blocked, clearer_side_speed(self._scan, self._env), self.velocity[:, 1]
            )

    def apply_actions(self) -> None:
        read_plan = self._env.scene.read_plan
        joint = read_plan.joint_tensor_view(self._entity)
        body = read_plan.body_tensor_view(self._entity, ("robot/pelvis",))
        pos_w, quat_w = body.pos_w[:, 0, :], body.quat_w[:, 0, :]
        ang_vel_b = world_to_body_vector(quat_w, body.ang_vel_w[:, 0, :])
        gravity = self._entity.data.gravity_vec_w(self._device)
        if gravity.ndim == 1:
            gravity = gravity[None, :]
        # projected_gravity_b = R^T g (upright -> (0,0,-1)), matching mjlab.
        gravity_b = world_to_body_vector(quat_w, gravity.expand(quat_w.shape[0], 3))
        default = self._entity.data.default_joint_pos(self._device)
        obs = self.policy.observe(
            ang_vel_b=ang_vel_b,
            gravity_b=gravity_b,
            joint_pos_rel=joint.joint_pos - default,
            joint_vel=joint.joint_vel,
            velocity=self.velocity,
            previous=self.previous_low,
            time_s=self.low_time,
        )
        self.previous_low.copy_(self.policy(obs))
        self.low_time += self.policy.control_dt
        self.low.process_actions(self.previous_low)
        self.low.apply_actions()

    def place_obstacles(self, env_ids: torch.Tensor | slice) -> None:
        """Teleport the obstacle boxes and plan the goal for the selected rows.

        Called from the reset event, inside the packed reset transaction (the
        action term's own ``reset`` runs after the transaction closes). The
        goal direction/distance sampled here is reused by ``reset`` so the
        boxes sit on the path to the goal, matching InstinctMJ's joint draw.
        """
        if isinstance(env_ids, slice):
            rows = torch.arange(self.num_envs, device=self._device)
        else:
            rows = torch.as_tensor(env_ids, device=self._device).reshape(-1)
        count = rows.numel()
        if count == 0:
            return
        device = self._device
        origin = self._env.scene.env_origins(device).index_select(0, rows)
        angle = torch.rand(count, device=device) * 1.2 - 0.6
        distance = torch.rand(count, device=device) * 2 + 3
        if self.stage == 3:
            distance = torch.rand(count, device=device) * 0.5 + 4.5
        self._planned_angle[rows] = angle
        self._planned_distance[rows] = distance
        choice = (
            torch.randint(0, 3, (count,), device=device)
            if self.stage >= 2
            else torch.ones(count, device=device, dtype=torch.long)
        )
        heights = (0.3, 0.45, 0.6)
        for index in range(3):
            pose = torch.zeros(count, 7, device=device)
            pose[:, 3] = 1
            along = torch.rand(count, device=device) * 0.8 + 1.2
            if self.stage == 3:
                along = torch.full_like(along, 1.0 + index)
            lateral = torch.rand(count, device=device) * 1.0 - 0.5
            pose[:, 0] = origin[:, 0] + along * angle.cos() - lateral * angle.sin()
            pose[:, 1] = origin[:, 1] + along * angle.sin() + lateral * angle.cos()
            pose[:, 2] = heights[index] / 2
            active = (
                torch.ones_like(choice, dtype=torch.bool)
                if self.stage == 3
                else choice == index
            )
            if self.stage == 0:
                active[:] = False
            pose[~active, 2] = -10
            obstacle, body_name = self._mocap_entities[index]
            obstacle.write_mocap_pose_to_sim(
                pose.detach().cpu().numpy(), env_ids=rows, term_name="navigation"
            )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.velocity[ids] = 0
        self.previous_low[ids] = 0
        self.low_time[ids] = 0
        self.best_distance[ids] = 0
        self.progress[ids] = 0
        self.interval_progress[ids] = 0
        self.low.reset(ids)
        if isinstance(ids, slice):
            rows = torch.arange(self.num_envs, device=self._device)
        else:
            rows = torch.as_tensor(ids, device=self._device).reshape(-1)
        count = rows.numel()
        if count == 0:
            self._high_level_counter = 0
            return
        device = self._device
        origin = self._env.scene.env_origins(device).index_select(0, rows)
        # The reset event planned these together with the obstacle poses.
        angle = self._planned_angle[rows]
        distance = self._planned_distance[rows]
        self.goal[rows, :2] = origin[:, :2] + torch.stack(
            (distance * angle.cos(), distance * angle.sin()), -1
        )
        self.goal[rows, 2] = 0
        self.best_distance[rows] = distance
        self.reward_step = -1
        if self.cfg.log_goal:
            goals = self.goal[rows].detach().cpu()
            distances = self.best_distance[rows].detach().cpu()
            for goal, goal_distance in zip(goals, distances, strict=True):
                print(
                    f"goal: x={float(goal[0]):+.3f} y={float(goal[1]):+.3f} "
                    f"z={float(goal[2]):+.3f} distance={float(goal_distance):.3f}"
                )
        self._high_level_counter = 0


class NavigationManagerBasedEnv(_ConcreteManagerBasedRlEnv):
    """Manager-Based navigation runtime with the play keyboard hook."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_manager.get_term("navigation").bind_mocap_bodies()

    def get_playback_debug_overlays(self):
        """Include action-term overlays (the base aggregates command terms only)."""
        for name in self.action_manager.active_terms:
            provider = getattr(self.action_manager.get_term(name), "playback_debug_overlay_getter", None)
            if provider is not None:
                getter = provider()
                if getter is not None:
                    return getter
        return None

    def play_key_hook(self, keycode: int) -> bool:
        """Consume teleop keys; True when the keycode was a navigation key."""
        try:
            term = self.action_manager.get_term("navigation")
        except KeyError:
            return False
        if not isinstance(term, NavigationAction) or not term.teleop_active:
            return False
        if keycode == _GLF_KEY_UP:
            term.nudge_teleop(vx=0.1)
        elif keycode == _GLF_KEY_DOWN:
            term.nudge_teleop(vx=-0.1)
        elif keycode == _GLF_KEY_LEFT:
            term.nudge_teleop(vyaw=0.1)
        elif keycode == _GLF_KEY_RIGHT:
            term.nudge_teleop(vyaw=-0.1)
        elif keycode in (_GLF_KEY_ENTER, 335):  # Enter / KP_Enter
            term.clear_teleop()
        else:
            return False
        return True


def make_navigation_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> NavigationManagerBasedEnv:
    """Construct the Registry-owned navigation Manager-Based runtime."""
    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError(
            f"make_navigation_env expected ManagerBasedRlEnvCfg, received {type(cfg).__name__}"
        )
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError(f"make_navigation_env num_envs must be a positive integer, got {num_envs!r}")
    if not isinstance(backend_type, str) or not backend_type:
        raise ValueError(f"make_navigation_env backend_type must be a non-empty string, got {backend_type!r}")

    cfg.validate()
    assert cfg.scene is not None
    # InstinctMJ teleop play semantics: teleop replaces the task objective, so
    # all terminations are disabled (play.py: `env_cfg.terminations = {}`) and
    # episodes never reset while the user is driving.
    action_cfg = (cfg.actions or {}).get("navigation")
    if action_cfg is not None and bool(getattr(action_cfg, "teleop", False)):
        cfg.terminations = {}
        cfg.validate()
    base_name, body_state_requested, tracked_body_names = _resolve_backend_entity_contract(cfg)
    backend_kwargs = env_backend_kwargs(cfg)
    backend_kwargs["base_name"] = base_name
    if backend_type == "mujoco" and tracked_body_names is not None:
        backend_kwargs["tracked_body_names"] = tracked_body_names

    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=body_state_requested,
        **backend_kwargs,
    )
    try:
        return NavigationManagerBasedEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


__all__ = [
    "NavigationAction",
    "NavigationActionCfg",
    "NavigationManagerBasedEnv",
    "make_navigation_env",
]