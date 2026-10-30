"""Navigation control/reward regressions without a simulator or ONNX inference."""

from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
import torch
from hydra import compose, initialize_config_dir

from unilab.managers.manager_base import ManagerTermBaseCfg
from unilab.tasks.navigation import manager_terms, mdp


class Scene(dict):
    read_plan = object()

    def env_origins(self, device):
        return torch.zeros(2, 3, device=device)


@pytest.fixture
def navigation(monkeypatch):
    names = tuple(f"robot/{name}" for name in manager_terms.OFFICIAL_JOINT_ORDER)
    scene = Scene(robot=SimpleNamespace(joint_names=names))
    for index in range(3):
        scene[f"obstacle_{index}"] = SimpleNamespace()
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.02, physics_dt=0.005, scene=scene)
    limits = torch.tensor([[-1.0, 1.0]] * 3)
    monkeypatch.setattr(
        manager_terms,
        "UnitreeOnnxLocomotion",
        lambda *args, **kwargs: SimpleNamespace(
            control_dt=0.02,
            limits=limits,
            command=lambda nominal, residual: (nominal + residual).clamp(-1, 1),
        ),
    )
    monkeypatch.setattr(
        manager_terms.JointPositionActionCfg,
        "build",
        lambda *args: SimpleNamespace(action_dim=29, target_names=names, reset=lambda ids: None),
    )
    monkeypatch.setattr(manager_terms, "NormalizedRayScan", lambda *args: None)
    term = manager_terms.NavigationActionCfg(entity_name="robot").build(env)
    monkeypatch.setattr(term, "relative_goal", lambda: torch.zeros(2, 3))
    env.action_manager = SimpleNamespace(
        get_term=lambda name: term, action=torch.zeros(2, 3), prev_action=torch.zeros(2, 3)
    )
    env.common_step_counter = 0
    return env, term


def reward(env, kind):
    return mdp.navigation_reward(
        ManagerTermBaseCfg(func=mdp.navigation_reward, params={"kind": kind}), env
    )


def test_every_policy_action_changes_the_executed_command(navigation):
    _, term = navigation
    for index in range(10):
        actions = torch.full((2, 3), (index - 5) / 10)
        term.process_actions(actions)
        torch.testing.assert_close(term.velocity, actions)
        torch.testing.assert_close(term.raw_action, actions)


def test_partial_reset_preserves_other_rows_and_next_action_is_immediate(navigation):
    _, term = navigation
    term.process_actions(torch.full((2, 3), 0.4))
    term._planned_distance[:] = 4
    term.obstacle_contact[:] = True
    term.reset(torch.tensor([0]))
    assert term.obstacle_contact.tolist() == [False, True]
    torch.testing.assert_close(term.velocity[0], torch.zeros(3))
    torch.testing.assert_close(term.velocity[1], torch.full((3,), 0.4))
    actions = torch.tensor([[0.2, 0.1, -0.1], [-0.2, 0.3, 0.4]])
    term.process_actions(actions)
    assert not term.obstacle_contact.any()
    torch.testing.assert_close(term.velocity, actions)


@pytest.mark.parametrize("kind,weight", [("collision", -10), ("fall", -10), ("success", 10)])
def test_terminal_reward_is_paid_at_every_control_step(navigation, kind, weight):
    env, term = navigation
    outcome_reward = reward(env, kind)
    outcome_reward._outcome = lambda env: torch.tensor([True, False])
    # Exercise all five time positions that previously gated terminal rewards.
    for index in range(5):
        env.common_step_counter = index + 1
        term.process_actions(torch.zeros(2, 3))
        actual = outcome_reward(env) * env.step_dt * weight
        torch.testing.assert_close(actual, torch.tensor([float(weight), 0.0]))


def test_progress_is_paid_immediately_and_backtracking_cannot_earn_it_again(navigation):
    env, term = navigation
    term.best_distance[:] = 4
    distance = torch.full((2,), 4.0)
    term.distance = lambda: distance
    progress = reward(env, "progress")
    for index, (value, expected) in enumerate([(3.9, 0.1), (4.0, 0), (3.9, 0), (3.7, 0.2)]):
        env.common_step_counter = index + 1
        distance.fill_(value)
        actual = progress(env) * env.step_dt
        torch.testing.assert_close(actual, torch.full((2,), float(expected)), atol=1e-6, rtol=0)
        # Re-reading a term does not consume or advance its progress bookkeeping.
        torch.testing.assert_close(progress(env) * env.step_dt, actual)


def test_time_cost_and_action_ramp_cost_preserve_physical_scale(navigation):
    env, _ = navigation
    time = reward(env, "time")
    smoothness = reward(env, "smoothness")
    accumulated = torch.zeros(2)
    for index in range(5):
        env.action_manager.prev_action[:] = index * 0.1
        env.action_manager.action[:] = (index + 1) * 0.1
        accumulated += smoothness(env) * env.step_dt * -0.05
    # Same linear ramp from 0 to .5 in .1 s as one original 10 Hz decision.
    torch.testing.assert_close(accumulated, torch.full((2,), 3 * 0.5**2 * 0.1 * -0.05))
    torch.testing.assert_close(time(env) * 50 * env.step_dt * -0.1, torch.full((2,), -0.1))


def test_owner_preserves_rollout_and_discount_time_scales():
    config_dir = Path(__file__).resolve().parents[2] / "src/unilab/conf/ppo"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_navigation_raycast_unitree/mujoco"])
    assert cfg.env.ctrl_dt == pytest.approx(0.02)
    assert cfg.algo.num_steps_per_env * cfg.env.ctrl_dt == pytest.approx(2.4)
    assert cfg.algo.algorithm.gamma**5 == pytest.approx(0.99)
    assert cfg.algo.algorithm.lam**5 == pytest.approx(0.95)
    assert "high_level_every" not in cfg.env.actions.navigation


@pytest.mark.slow
@pytest.mark.parametrize(
    "offset",
    [(0.01, 0, 0.15), (0, 0, -0.08), (-0.03, 0.1, -0.2), (0, 0.1, -0.45), (0.05, 0.1, -0.75)],
    ids=["torso", "pelvis", "thigh", "shin", "foot"],
)
def test_mujoco_collision_penalty_and_autoreset_at_each_control_phase(offset):
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    from unilab.base.config_adapter import EnvConfigAdapter, create_env

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=g1_navigation_raycast_unitree/mujoco",
                "algo.num_envs=2",
                "training.device=cpu",
            ],
        )
    overrides = EnvConfigAdapter(cfg, root_dir=root).build_task_env_cfg_override()
    overrides["collector_tensor_device"] = "cpu"
    env = create_env(cfg, num_envs=2, env_cfg_override=overrides)
    try:
        for phase in range(5):
            env.reset()
            for _ in range(phase):
                env.step(torch.zeros(2, 3))
            env.scene.read_plan.refresh()
            pelvis = env.scene.read_plan.body_tensor_view(env.scene["robot"], ("robot/pelvis",))
            pose = torch.zeros(1, 7)
            pose[0, :3] = pelvis.pos_w[0, 0] + torch.tensor(offset)
            pose[0, 3] = 1
            # Test-only contact injection through the ordinary owner transaction;
            # leave row 1 untouched and do not reset the policy's action history.
            with env._reset_state.scoped_device_owner_tensor(torch.tensor([True, False])):
                env.scene["obstacle_0"].write_mocap_pose_to_sim(
                    pose.numpy(), env_ids=torch.tensor([0]), term_name="navigation"
                )
            state = env.step(torch.zeros(2, 3))
            assert state.terminated.tolist() == [True, False]
            assert state.info["log"]["Episode_Termination/collision"] == 1
            assert state.reward[0] < -9  # Full -10 collision cost plus small dense rewards.
            assert state.info["steps"][0] == 0  # Autoreset has already reset this row.
            assert all(torch.isfinite(value).all() for value in state.obs.values())
            # Contact history was cleared only for the automatically reset row;
            # stale contact must not cause a second collision in the next episode.
            state = env.step(torch.zeros(2, 3))
            assert not state.terminated.any()
    finally:
        env.close()


def test_contact_sensor_inventory_covers_every_robot_geom_and_only_obstacles():
    assets = Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/g1/navigation_unitree"
    robot = ElementTree.parse(assets / "g1.xml")
    geoms = {
        geom.attrib["name"]
        for geom in robot.findall(".//worldbody//geom")
        if geom.get("class") in {"collision", "foot_capsule"}
    }
    assert geoms == {f"{name}_collision" for name in mdp.ROBOT_COLLISION_GEOMS}
    sensors = ElementTree.parse(assets / "navigation_sensors.xml").findall("sensor/contact")
    assert len(sensors) == len(geoms) * 3 * 2
    assert {sensor.attrib["name"] for sensor in sensors} == set(mdp.OBSTACLE_CONTACT_SENSORS)
    for index in range(3):
        for field in ("found", "force"):
            matching = [
                sensor for sensor in sensors
                if sensor.get("geom2") == f"obstacle_{index}/obstacle_box_{index}_collision"
                and sensor.get("data") == field
            ]
            assert {sensor.get("geom1") for sensor in matching} == {f"robot/{name}" for name in geoms}


@pytest.mark.parametrize("geom", mdp.ROBOT_COLLISION_GEOMS)
@pytest.mark.parametrize("field", ["found", "force"])
def test_transient_contact_survives_later_clear_substeps(navigation, geom, field):
    env, term = navigation
    values = {
        name: torch.zeros(2, 3 if name.endswith("force") else 1)
        for name in mdp.OBSTACLE_CONTACT_SENSORS
    }
    env.scene.read_plan = SimpleNamespace(
        invalidate=lambda: None,
        refresh=lambda: None,
        sensor_tensor_views=lambda *args: SimpleNamespace(values=values),
    )
    calls = []

    def step_tensor(ctrl, nsteps):
        calls.append((ctrl.clone(), nsteps))
        for value in values.values():
            value.zero_()
        if len(calls) == 1:
            values[f"obstacle_contact_2_{geom}_{field}"][0, 0] = 1
        return {"timing": {"physics_ms": 0.5}}

    owner = manager_terms.NavigationManagerBasedEnv.__new__(manager_terms.NavigationManagerBasedEnv)
    owner._cfg = SimpleNamespace(sim_substeps=4)
    owner._backend = SimpleNamespace(step_tensor=step_tensor)
    owner._num_envs = 2
    owner._device = torch.device("cpu")
    owner.scene = env.scene
    owner.action_manager = env.action_manager
    ctrl = torch.ones(2, 29)
    result = owner.advance_physics(ctrl)
    assert not mdp.read_obstacle_contacts(owner).any()  # Final sample has no contact.
    assert term.obstacle_contact.tolist() == [True, False]
    assert result["timing"]["physics_ms"] == 2
    assert len(calls) == 4
    for actual, nsteps in calls:
        assert nsteps == 1
        torch.testing.assert_close(actual, ctrl)
