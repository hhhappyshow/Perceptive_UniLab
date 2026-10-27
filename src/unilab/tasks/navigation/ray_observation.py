"""Navigation forward-scan observation on UniLab's batch raycaster.

Ports InstinctMJ ``instinct_mj.perception.forward_scan`` to UniLab 1.5.2's
``RayScan`` observation term. Geometry: 16x32 pinhole rays on the torso, +X
forward, +Y left, pitched down 15 degrees, 3 m range. InstinctMJ normalized
distances by ``MAX_DISTANCE`` and encoded invalid/zero readings as the range
maximum; ``NormalizedRayScan`` keeps that contract on top of ``RayScan``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from unilab.envs.mdp.ray_observation import RayPatternCfg, RayCameraPatternCfg, RayScan

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

MAX_DISTANCE = 3.0
HEIGHT, WIDTH = 16, 32
PITCH_DOWN_DEG = 15.0

# InstinctMJ _FRONT_ROWS/_FRONT_COLS: the central window checked for a blocking
# box in front of the robot.
FRONT_ROWS = slice(9, 14)
FRONT_COLS = slice(8, 24)
BLOCK_DISTANCE = 0.8


def make_forward_scan_pattern() -> RayPatternCfg:
    """16x32 torso pinhole pattern pitched down 15 degrees.

    ``RayCameraPatternCfg`` requires orthogonal ``forward``/``up``; pitching is
    expressed by tilting both basis vectors instead of post-rotating rays like
    InstinctMJ's ``ForwardPattern.generate_rays``. Columns still run left to
    right and rows top to bottom, so InstinctMJ's row/col windows carry over.
    """
    pitch = math.radians(PITCH_DOWN_DEG)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    pattern = RayCameraPatternCfg(
        width=WIDTH,
        height=HEIGHT,
        fovy=45.0,
        forward=(cos_p, 0.0, -sin_p),
        up=(sin_p, 0.0, cos_p),
    )
    return RayPatternCfg(
        entity_name="robot",
        pattern=pattern,
        max_distance=MAX_DISTANCE,
        attachment_body_name="robot/torso_link",
        alignment="yaw",
        caster="uni_ray",
        allowed_geom_names=(
            "floor/floor",
            "obstacle_0/obstacle_box_0_collision",
            "obstacle_1/obstacle_box_1_collision",
            "obstacle_2/obstacle_box_2_collision",
        ),
    )


class NormalizedRayScan(RayScan):
    """RayScan with InstinctMJ's observation normalization.

    Raw ray distances are clamped to ``[0, max_distance]``; misses (the
    caster's out-of-range/invalid encoding, including zero readings) map to
    ``max_distance``. The returned observation is distance / max_distance,
    matching InstinctMJ's ``forward_scan(...)['ray_observation']``.
    """

    _trace_cache_counter: int = -1
    _trace_cache: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        ray_pattern: RayPatternCfg | None = None,
    ) -> torch.Tensor:
        distances = self._trace_once(env, ray_pattern)
        max_distance = self.ray_pattern.max_distance
        valid = torch.isfinite(distances) & (distances > 0.0) & (distances <= max_distance)
        distances = torch.where(valid, distances, torch.full_like(distances, max_distance))
        return (distances / max_distance).reshape(env.num_envs, -1)

    def _trace_once(
        self,
        env: ManagerBasedRlEnv,
        ray_pattern: RayPatternCfg | None = None,
    ) -> torch.Tensor:
        """Trace at most once per env step and share the raw result.

        The navigation action term's obstacle rule and the policy observation
        read the same step's scan without a second caster pass.
        """
        counter = getattr(env, "common_step_counter", None)
        if isinstance(counter, int) and self._trace_cache_counter == counter:
            assert self._trace_cache is not None
            return self._trace_cache
        raw = super().__call__(env, ray_pattern)
        distances = raw[..., 0] if raw.ndim == 3 else raw
        if isinstance(counter, int):
            self._trace_cache_counter = counter
            self._trace_cache = distances
        return distances

    def raw_distances(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """Un-normalized clamped distances in meters, shape ``(N, rays)``.

        Shared by the navigation action term's obstacle rule so the play-time
        safety layer and the policy observation read one trace.
        """
        max_distance = self.ray_pattern.max_distance
        distances = self._trace_once(env)
        valid = torch.isfinite(distances) & (distances > 0.0) & (distances <= max_distance)
        return torch.where(valid, distances, torch.full_like(distances, max_distance))

    def ensure_bound(self, env: ManagerBasedRlEnv) -> None:
        """Bind the caster on first use (the action term owns its own instance)."""
        if not self._bound and not self._closed:
            self.bind_read_phase()

    def ray_debug(self, env: ManagerBasedRlEnv) -> tuple["np.ndarray", "np.ndarray"]:
        """World-frame ray origins and hit endpoints for playback visualization.

        Returns two ``(num_envs, ray_count, 3)`` float arrays; endpoints use
        the clamped distances (misses reach ``max_distance``), mirroring
        mjlab's ray visualization. Requires one ``ensure_bound``/trace first.
        """
        import numpy as np

        distances = self.raw_distances(env).detach().cpu().numpy()
        origins = self._ray_origins.detach().cpu().numpy()
        directions = self._ray_directions.detach().cpu().numpy()
        endpoints = origins + directions * distances[:, :, None]
        return origins, endpoints

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Retain caster state; drop the per-step trace cache."""
        self._trace_cache = None
        self._trace_cache_counter = -1
        super().reset(env_ids)


def _view(distances: torch.Tensor) -> torch.Tensor:
    """Row-major (N, HEIGHT, WIDTH) ray-distance image."""
    return distances.view(distances.shape[0], HEIGHT, WIDTH)


def forward_blocked(scan: Any, env: ManagerBasedRlEnv) -> torch.Tensor:
    """True where the nearest central ray meets a box closer than 0.8 m."""
    image = _view(scan.raw_distances(env))
    nearest = image[:, FRONT_ROWS, FRONT_COLS].amin(dim=(1, 2))
    return (nearest > 0.05) & (nearest < BLOCK_DISTANCE)


def clearer_side_speed(scan: Any, env: ManagerBasedRlEnv) -> torch.Tensor:
    """+0.4 m/s toward the half-image whose forward rays reach farther."""
    image = _view(scan.raw_distances(env))
    band = image[:, FRONT_ROWS, :]
    left = band[:, :, : WIDTH // 2].amin(dim=(1, 2))
    right = band[:, :, WIDTH // 2 :].amin(dim=(1, 2))
    return torch.where(left >= right, torch.full_like(left, 0.4), torch.full_like(left, -0.4))


__all__ = [
    "BLOCK_DISTANCE",
    "FRONT_COLS",
    "FRONT_ROWS",
    "HEIGHT",
    "MAX_DISTANCE",
    "WIDTH",
    "NormalizedRayScan",
    "clearer_side_speed",
    "forward_blocked",
    "make_forward_scan_pattern",
]