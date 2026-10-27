"""UniLab Manager-Based G1 navigation (raycast obstacle avoidance) task.

Port of InstinctMJ's ``Instinct-G1-RayCaster-Obstacle-Avoidance-Unitree-v0``
(Stage 1, official unitree ONNX low-level backend). The env declaration lives
in ``conf/ppo/task/g1_navigation_raycast_unitree``; the action term, mdp
terms, and the MuJoCo scene file are task-owned here.
"""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .manager_terms import NavigationManagerBasedEnv, make_navigation_env

registry.register_env_config("G1NavigationRayCastUnitree", ManagerBasedRlEnvCfg)
registry.register_env("G1NavigationRayCastUnitree", make_navigation_env, sim_backend="mujoco")

__all__ = [
    "NavigationManagerBasedEnv",
    "make_navigation_env",
]