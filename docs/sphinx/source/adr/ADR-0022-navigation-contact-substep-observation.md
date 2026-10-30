---
orphan: true
---

# ADR-0022 导航任务物理子步接触记录

- Status: Proposed
- Date: 2026-10-30
- Owners: UniLab maintainers
- Supersedes: None
- Superseded by: None

## Context

导航任务需要将机器人全身与障碍物的接触纳入失败判定。只读取控制周期末尾的
contact sensor 会漏掉前面物理子步已经结束的接触。当前 UniSim 实体场景的传感器
片段不支持 MuJoCo 的 `delay`、`nsample` 或 `subtree1` 属性；tensor stepping
也不支持 host pre-step callback。

## Decision

在 TorchEnv 增加 `advance_physics(ctrl)` 扩展点，默认仍调用一次公开的
`SimBackend.step_tensor(ctrl, sim_substeps)`。导航的 Manager-Based 子类逐物理
子步调用该公开接口，以相同控制推进，并从 packed read plan 读取接触，按环境行
累计逻辑或。不修改依赖包，不读取后端私有物理数据。

## Stable Contracts

- 动作处理、奖励、episode 计数及自动重置仍每个控制步执行一次。
- 导航传感器明确配对所有机器人碰撞几何体和三个障碍物，不包含地面或自碰撞。
- 每步动作开始时清空接触记录，每个物理子步累积；控制步末统一判定碰撞并扣分。
- 选中行重置仅清除选中行的记录。导航当前仅支持同步 MuJoCo 路径。
- 默认环境保留原来的单次批量步进；扩展点返回该控制周期的后端结果，完成回执
  必须覆盖整个周期。导航累计各子步 timing，不改变控制周期时长。

## Alternatives Considered

- MuJoCo 原生延迟传感器：当前公开场景组装合同不支持，不能直接透传。
- 控制步末读取当前接触：无法恢复已经消失的接触。
- 每个物理子步调用完整 env.step：会改变 PPO 决策频率和自动重置语义。

## Consequences

导航每个控制周期增加物理调用和 packed read 次数，有性能成本，不能声称吞吐不变。
这记录离散物理子步上已被仿真器识别的接触，不提供连续碰撞检测保证。

## Evidence In Repo

- `src/unilab/base/torch_env.py`
- `src/unilab/tasks/navigation/manager_terms.py`
- `src/unilab/tasks/navigation/mdp.py`
- `src/unilab/assets/robots/g1/navigation_unitree/navigation_sensors.xml`
- `tests/tasks/test_navigation.py`
- `tests/base/test_torch_env.py`

## Related Documents

- {doc}`ADR-0012 </adr/ADR-0012-sole-tensor-manager-and-scoped-backends>`
- {doc}`ADR Index </adr/README>`
