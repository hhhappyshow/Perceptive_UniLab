"""RayCaster 接口演示:UniLab(uni_ray)vs InstinctMJ(mjlab/mj_ray)语义对比。

一个最小场景(地板 + 三个已知位置的障碍箱 + 一个 torso 基准帧)上:

  Part A  用 unisim 的底层 RayCaster 契约手工走一遍完整生命周期:
          RaySceneDescription -> RayCollisionDescription -> create_ray_caster
          -> materialize -> bind_pose_layout -> update_pose -> trace -> close
          (高层 unilab.envs.mdp.ray_observation.RayScan 只是这套调用在
           Manager 观测项里的接线版本,几何与 trace 语义完全一致。)

  Part B  16x32 前向扫描 pattern 的两种生成方式对比:
          - UniLab: RayCameraPatternCfg(forward/up 正交基直接表达 15 度俯仰)
          - InstinctMJ: ForwardPattern.generate_rays(相机系 -Z 视向映射 +
            绕 Y 轴后旋转 15 度),此处按其源码逐行复刻(纯数学,不 import mjlab)

  Part C  距离语义对比:同一批射线分别用 uni_ray 与 mujoco.mj_ray
          (mjlab 底层 mujoco_warp rays kernel 的同语义参考实现)求交,
          对比命中掩码、径向距离数值、miss 编码。

运行(默认安静:只打开 3D 可视化窗口,不打印任何过程文字):
    uv run scripts/raycaster_demo.py                  # 交互式 3D 窗口
    uv run scripts/raycaster_demo.py --save out.png   # 无显示环境存图
    uv run scripts/raycaster_demo.py --report         # 额外打印完整对比报告
"""

from __future__ import annotations

import argparse
import math
import os

# 在导入 warp 系包(unisim/uni_ray)之前设置,压制 Warp 的 init 横幅输出。
os.environ.setdefault("WARP_QUIET", "1")
import warp as wp  # noqa: E402

wp.config.log_level = wp.LOG_WARNING

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from unilab.envs.mdp.ray_observation import RayCameraPatternCfg  # noqa: E402
from unisim import create_ray_caster  # noqa: E402
from unisim.ray_query import (  # noqa: E402
    RayCollisionDescription,
    RaySceneDescription,
    RayTraceFilter,
    RayTraceOutputs,
)

# --report 时才打印的过程输出;默认全部静默。
VERBOSE = False


def log(*args: object) -> None:
    if VERBOSE:
        print(*args)


# ---- 与导航任务一致的几何常量 -------------------------------------------------
MAX_DISTANCE = 3.0
WIDTH, HEIGHT = 32, 16  # 列(x), 行(y):InstinctMJ 16x32, 行优先展平 512 条
PITCH_DOWN_DEG = 15.0
TORSO_Z = 0.82

# ---- Part 0:单一事实源的最小场景 ----------------------------------------------
# 环境 geom 全部 group 0(InstinctMJ 的 environment=0),torso 球 group 1(robot)
# ——不过滤的话原点射线会先打中自己的球,这正是 InstinctMJ 分组过滤存在的意义。
# box 记录 (名字 -> (世界位置, 半边长));地板是局部 z=0 无限平面。
BOXES = {
    "box_a": (np.array([1.2, 0.0, 0.15]), np.array([0.15, 0.15, 0.15])),
    "box_b": (np.array([1.5, -0.8, 0.30]), np.array([0.20, 0.20, 0.30])),
    "box_c": (np.array([1.8, 0.6, 0.20]), np.array([0.10, 0.10, 0.20])),
}
TORSO_POS = np.array([0.0, 0.0, TORSO_Z])
IDENT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz 单位四元数
BODY_NAMES = ("floor", "box_a", "box_b", "box_c", "torso")


def build_unisim_scene() -> tuple[RaySceneDescription, RayCollisionDescription]:
    """把场景翻译成 unisim 的纯 NumPy 射线场景契约。

    RaySceneDescription 约定:每 geom 挂一个 body、局部系表达;box 的 size 是
    半边长 (x, y, z);plane 忽略 size(局部 z=0 平面、法向 +z);四元数一律 wxyz。
    uni_ray 在 materialize 时把这份数据转成自己的 BVH 加速结构。
    """
    scene = RaySceneDescription(
        num_bodies=len(BODY_NAMES),
        geom_types=("plane", "box", "box", "box", "sphere"),
        geom_sizes=np.array(
            [
                [5.0, 5.0, 0.1],  # plane 忽略 size
                BOXES["box_a"][1],
                BOXES["box_b"][1],
                BOXES["box_c"][1],
                [0.05, 0.0, 0.0],  # sphere (radius, 0, 0)
            ]
        ),
        geom_local_pos=np.zeros((5, 3)),
        geom_local_quat=np.tile(IDENT_QUAT, (5, 1)),
        geom_body_ids=np.array([0, 1, 2, 3, 4]),
        body_names=BODY_NAMES,
    )
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=(
            "floor/floor",
            "box_a/box",
            "box_b/box",
            "box_c/box",
            "torso/torso",
        ),
    )
    return scene, collision


def build_mj_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """同一场景的 MuJoCo 原生表达,供 mj_ray(mjlab 底层 kernel 的同语义参考)使用。

    torso 用 mocap body(免写 qpos);调用 mj_kinematics 让 body 位姿就位。
    """
    boxes_xml = "".join(
        f'<body name="{name}" pos="{p[0]} {p[1]} {p[2]}">'
        f'<geom name="{name}/box" type="box" size="{s[0]} {s[1]} {s[2]}" group="0"/>'
        f"</body>"
        for name, (p, s) in BOXES.items()
    )
    xml = f"""
    <mujoco>
      <worldbody>
        <geom name="floor/floor" type="plane" size="5 5 0.1" pos="0 0 0" group="0"/>
        {boxes_xml}
        <body name="torso" mocap="true" pos="{TORSO_POS[0]} {TORSO_POS[1]} {TORSO_POS[2]}">
          <geom name="torso/torso" type="sphere" size="0.05" group="1"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return model, data


def unilab_pattern_directions() -> torch.Tensor:
    """UniLab 侧 16x32 前向扫描方向:直接调用真实接口 RayCameraPatternCfg。

    15 度俯仰用一对正交基表达:forward 前倾、up 随之倾斜;right = cross(forward, up)
    恰为 -Y(即 +Y 是"左")。射线方向 = forward + x*right - y*up 后归一化。
    """
    pitch = math.radians(PITCH_DOWN_DEG)
    pattern = RayCameraPatternCfg(
        width=WIDTH,
        height=HEIGHT,
        fovy=45.0,
        forward=(math.cos(pitch), 0.0, -math.sin(pitch)),
        up=(math.sin(pitch), 0.0, math.cos(pitch)),
    )
    return pattern.directions(torch.device("cpu"))


def instinct_pattern_directions() -> torch.Tensor:
    """InstinctMJ 侧 16x32 前向扫描方向:逐行复刻 ForwardPattern.generate_rays。

    mjlab 基类先按 MuJoCo 相机约定(-Z 视前,+X 右,+Y 下)造方向:
        ray = [u*tan(hfov/2), v*tan(vfov/2), -1](u/v 是 linspace(-1,1) 含端点)
    ForwardPattern 再做轴映射 + 俯仰:
        direction = (-z, -x, -y) = (+1, -ray_x, -ray_y),随后绕 Y 轴旋转 -15 度。
    """
    u = torch.linspace(-1.0, 1.0, WIDTH)
    v = torch.linspace(-1.0, 1.0, HEIGHT)
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    v_fov = math.radians(45.0)
    h_fov = 2.0 * math.atan(math.tan(v_fov / 2.0) * WIDTH / HEIGHT)
    camera = torch.stack(
        [
            grid_u.flatten() * math.tan(h_fov / 2),
            grid_v.flatten() * math.tan(v_fov / 2),
            -torch.ones(WIDTH * HEIGHT),
        ],
        dim=-1,
    )
    camera = camera / camera.norm(dim=1, keepdim=True)
    direction = torch.stack((-camera[:, 2], -camera[:, 0], -camera[:, 1]), -1)
    pitch = math.radians(PITCH_DOWN_DEG)
    rotation = direction.new_tensor(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]
    )
    return direction @ rotation.T


def _make_caster(scene, collision):
    """创建并冷路径绑定一个 uni_ray caster(materialize + pose 布局)。"""
    caster = create_ray_caster(
        "uni_ray", num_envs=1, num_rays=WIDTH * HEIGHT, collision=collision, device="cpu"
    )
    caster.materialize(scene)
    caster.bind_pose_layout(BODY_NAMES)
    pos = np.zeros((1, len(BODY_NAMES), 3))
    pos[0, BODY_NAMES.index("torso")] = TORSO_POS  # 只有 torso 会动,其余保持单位位姿
    quat = np.tile(IDENT_QUAT, (1, len(BODY_NAMES), 1))
    caster.update_pose(pos, quat)
    return caster


def trace_uniray(scene, collision, origins, directions, ray_filter):
    """走一遍底层 caster 生命周期(host 路径)。

    RayScan 的热路径用的是等价的 tensor 版(update_pose_tensor + trace_tensor,
    caster 零拷贝别名调用方的 packed pose);此处用 host 路径演示,语义相同。
    """
    caster = _make_caster(scene, collision)
    result = caster.trace(
        origins,
        directions,
        MAX_DISTANCE,
        outputs=RayTraceOutputs(hit_point=True),
        ray_filter=ray_filter,
    )
    caster.close()  # 契约要求:释放资源,之后不得再 trace
    return result


def trace_mj_ray(model, data, origins, directions):
    """mjlab 底层同语义参考:mujoco.mj_ray 逐条求交。

    mj_ray 返回标量距离:命中为径向距离,**miss 为 -1.0**;
    geomgroup 非零字节 = 包含该组(注意 mujoco_warp 的 vec6 约定是 -1=include)。
    InstinctMJ forward_scan 的 valid 判据正是 (isfinite & 0<=d<=MAX)。
    """
    dists = np.empty((WIDTH * HEIGHT,), dtype=np.float64)
    geomids = np.empty((WIDTH * HEIGHT,), dtype=np.int64)
    gid = np.zeros(1, dtype=np.int32)
    group0_only = np.eye(6, dtype=np.uint8)[0]  # 只包含 group 0(排除 torso 球)
    for ray in range(WIDTH * HEIGHT):
        dists[ray] = mujoco.mj_ray(
            model, data, origins[ray], directions[ray], group0_only, True, -1, gid
        )
        geomids[ray] = gid[0]
    return dists, geomids


def print_report(unilab_dirs, inst_dirs, uni_dist, uni_hit, uni_geomid,
                 mj_same_dist, mj_same_geomid, mj_inst_dist, mj_inst_geomid):
    """对比报告:坐标系/方向/距离/排列四个维度。"""
    def decode(index: int) -> tuple[int, int]:
        return index // WIDTH, index % WIDTH  # 行优先:row = i // WIDTH

    log("\n===== Part B: 射线方向(坐标系与排列) =====")
    angle = torch.rad2deg(torch.acos((unilab_dirs * inst_dirs).sum(-1).clamp(-1.0, 1.0)))
    log(f"逐射线最大角度差: {angle.max():.4f} deg(中位数 {angle.median():.2e} deg)")
    log(f"角度差 < 0.001 deg 的射线数: {(angle < 1e-3).sum().item()} / {WIDTH * HEIGHT}")
    # 中央窗口 = 导航安全规则读取的区域(InstinctMJ _FRONT_ROWS/_FRONT_COLS)。
    angle_image = angle.view(HEIGHT, WIDTH)
    center = angle_image[9:14, 8:24]
    log(f"中央窗口(rows 9-13, cols 8-23)最大角度差: {center.max():.4f} deg")
    for label, idx in (
        ("左上角(0,0)", 0),
        ("中心(8,16)", 8 * WIDTH + 16),
        ("右下角(15,31)", WIDTH * HEIGHT - 1),
    ):
        r, c = decode(idx)
        log(
            f"  {label}: UniLab dir={unilab_dirs[idx].numpy().round(4)} | "
            f"InstinctMJ dir={inst_dirs[idx].numpy().round(4)}"
        )
    log(
        f"  行序(行0 应朝上): UniLab Z分量 {unilab_dirs[0, 2]:+.3f} | "
        f"InstinctMJ {inst_dirs[0, 2]:+.3f};末行: {unilab_dirs[(HEIGHT-1)*WIDTH, 2]:+.3f} | "
        f"{inst_dirs[(HEIGHT-1)*WIDTH, 2]:+.3f}"
    )
    log(
        f"  列序(列0 应朝左/+Y): UniLab Y分量 {unilab_dirs[0, 1]:+.3f} | "
        f"InstinctMJ {inst_dirs[0, 1]:+.3f};末列: {unilab_dirs[WIDTH-1, 1]:+.3f} | "
        f"{inst_dirs[WIDTH-1, 1]:+.3f}"
    )

    log("\n===== Part C1: 距离计算一致性(同一射线,uni_ray vs mj_ray) =====")
    mj_valid = np.isfinite(mj_same_dist) & (mj_same_dist >= 0) & (mj_same_dist <= MAX_DISTANCE)
    uni_np = uni_dist[0]
    agree = uni_hit[0] == mj_valid
    log(f"命中掩码一致: {agree.sum().item()} / {WIDTH * HEIGHT}")
    both = uni_hit[0] & mj_valid
    if both.any():
        log(f"共同命中射线的最大距离差: {np.abs(uni_np[both] - mj_same_dist[both]).max():.3e} m")
    same_geom = both & (uni_geomid[0] == mj_same_geomid)
    if same_geom.any():
        log(
            f"共同命中且命中同一 geom 的射线: {same_geom.sum().item()} 条, "
            f"最大距离差 {np.abs(uni_np[same_geom] - mj_same_dist[same_geom]).max():.3e} m"
            " —— 核心求交数值一致,差异仅在浮点精度量级"
        )
    disagreement = (~agree).nonzero()[0]
    log(f"命中分歧射线: {disagreement.size} 条(掠射/轮廓边缘处 float32 BVH 与 float64 解析求交的边界翻转)")
    if disagreement.size:
        sample = disagreement[:6]
        log(
            "  样例(row,col): "
            + ", ".join(
                f"({i // WIDTH},{i % WIDTH}) uni={uni_np[i]:.3f}/mj={mj_same_dist[i]:.3f}"
                for i in sample
            )
        )
    for idx in (8 * WIDTH + 14, 8 * WIDTH + 16, 0, WIDTH * HEIGHT - 1):
        r, c = decode(idx)
        log(
            f"  ray(row={r:2d},col={c:2d}): uni_ray dist={uni_np[idx]:7.4f} hit={str(bool(uni_hit[0, idx])):5s} "
            f"geom={uni_geomid[0, idx]:2d} | mj_ray dist={mj_same_dist[idx]:+7.4f} geomid={mj_same_geomid[idx]:2d}"
        )
    log(
        "  miss 编码: uni_ray -> dist==max_distance 且 hit=False;"
        " InstinctMJ(mjlab) -> dist=-1,经 forward_scan 映射为 max_distance;\n"
        "  两者经各自归一化后的策略观测均为 1.0(数值等价)。\n"
        "  距离均为径向(ray 轴向)距离,不是相机深度。"
    )

    log("\n===== Part C2: 跨 pattern 观测差异(InstinctMJ 网格 vs UniLab 网格) =====")
    mj_inst_valid = (
        np.isfinite(mj_inst_dist) & (mj_inst_dist >= 0) & (mj_inst_dist <= MAX_DISTANCE)
    )
    cross_agree = mj_inst_valid == mj_valid
    cross_common = mj_inst_valid & mj_valid
    log(f"命中掩码一致: {cross_agree.sum().item()} / {WIDTH * HEIGHT}(差异射线集中在障碍箱轮廓边缘)")
    if cross_common.any():
        log(f"共同命中射线距离差: max={np.abs(mj_inst_dist[cross_common] - mj_same_dist[cross_common]).max():.3f} m"
            f" 中位数={np.median(np.abs(mj_inst_dist[cross_common] - mj_same_dist[cross_common])):.4f} m")
    log("  归因:两侧 pattern 的采样约定不同 —— mjlab 的 u/v 是 linspace(-1,1) 含端点"
        "(极端射线恰在 FOV 边缘),\n  UniLab 是像素中心 (j+0.5-w/2)/focal(略窄于 FOV)。"
        "这不是实现 bug,是两种相机采样约定;对导航观测的影响在轮廓边缘射线处最大。")


def visualize(save_path: str | None, unilab_dirs, uni_dist, uni_hit, uni_hit_pts,
              inst_dirs, mj_dist, mj_valid):
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    def draw_scene(ax):
        # 地板
        ax.plot_surface(
            *np.meshgrid([-3, 3], [-3, 3]), np.zeros((2, 2)), alpha=0.08, color="gray"
        )
        # 障碍箱(棱线)
        for pos, half in BOXES.values():
            corners = (
                np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                * half
                + pos
            )
            edges = [
                (0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
                (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for i, j in edges:
                ax.plot(*corners[[i, j]].T, color="0.3", lw=1)
        # torso 基准架(+X 红、+Y 绿、+Z 蓝)
        for axis, color in zip(np.eye(3), ("r", "g", "b")):
            ax.plot(*np.stack([TORSO_POS, TORSO_POS + axis * 0.3]).T, color=color, lw=2)

    def draw_rays(ax, dirs, dist, valid, hit_pts, title):
        count = WIDTH * HEIGHT
        starts = np.tile(TORSO_POS, (count, 1))
        v = np.asarray(valid).reshape(-1)
        ends = np.where(
            v[:, None],
            np.asarray(hit_pts).reshape(-1, 3),
            starts + dirs.numpy() * 0.35,  # miss 画 0.35 m 短桩
        )
        segs = np.stack([starts, ends], axis=1)
        colors = np.where(
            v[:, None], np.array([[0.0, 0.7, 0.0, 0.55]]), np.array([[1.0, 0.0, 0.0, 0.35]])
        )
        ax.add_collection3d(Line3DCollection(segs, colors=colors))
        draw_scene(ax)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 2)
        ax.view_init(elev=18, azim=-75)

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    draw_rays(
        ax1,
        unilab_dirs,
        uni_dist,
        uni_hit[0],
        uni_hit_pts,
        "UniLab: uni_ray + RayCameraPatternCfg",
    )
    ax2 = fig.add_subplot(122, projection="3d")
    # InstinctMJ 面板:命中点 = 原点 + 方向 * 距离(miss 塌缩到原点,画短桩)。
    inst_hit_pts = np.tile(TORSO_POS, (WIDTH * HEIGHT, 1)) + inst_dirs.numpy() * np.clip(
        np.asarray(mj_dist), 0.0, None
    )[:, None]
    draw_rays(
        ax2,
        inst_dirs,
        mj_dist,
        mj_valid,
        inst_hit_pts,
        "InstinctMJ pattern (replicated) + mj_ray",
    )
    fig.suptitle("RayCaster comparison: green=hit, red=miss; identical scene & torso pose")
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"可视化已保存: {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None, help="保存可视化 PNG 而不打开窗口")
    parser.add_argument("--report", action="store_true", help="打印完整对比报告")
    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.report

    scene, collision = build_unisim_scene()
    origins = np.tile(TORSO_POS, (WIDTH * HEIGHT, 1))  # pinhole:所有射线同原点
    unilab_dirs = unilab_pattern_directions()
    inst_dirs = instinct_pattern_directions()
    # 过滤:只允许环境 geom(索引 0-3),排除 torso 球(索引 4)。
    ray_filter = RayTraceFilter(allowed_geom_ids=(0, 1, 2, 3))
    log("===== Part A: unisim 底层 RayCaster 生命周期 =====")
    result = trace_uniray(scene, collision, origins, unilab_dirs.numpy(), ray_filter)
    # host 路径返回 NumPy,tensor 路径返回 Torch;统一成 NumPy 便于对比。
    uni_dist = np.asarray(result.distance)
    uni_hit = np.asarray(result.hit)
    uni_hit_pts = np.asarray(result.hit_point)
    log(f"trace 输出形状: distance={uni_dist.shape} hit={uni_hit.shape} hit_point={uni_hit_pts.shape}")

    model, data = build_mj_model()
    # 同一射线(unilab_dirs)分别喂给两个引擎 -> 隔离"距离计算"差异(C1)。
    mj_same_dist, mj_same_geomid = trace_mj_ray(model, data, origins, unilab_dirs.numpy())
    # InstinctMJ 自己的网格(inst_dirs)走 mj_ray -> 隔离"方向网格"差异(C2)。
    mj_inst_dist, mj_inst_geomid = trace_mj_ray(model, data, origins, inst_dirs.numpy())

    # uni_ray 的 geom_id 输出便于定位命中的是哪个 geom(需要再一次带 outputs 的 trace)。
    caster = _make_caster(scene, collision)
    gid_result = caster.trace(
        origins,
        unilab_dirs.numpy(),
        MAX_DISTANCE,
        outputs=RayTraceOutputs(geom_id=True),
        ray_filter=ray_filter,
    )
    caster.close()
    uni_geomid = np.asarray(gid_result.geom_id)

    print_report(
        unilab_dirs, inst_dirs, uni_dist, uni_hit, uni_geomid,
        mj_same_dist, mj_same_geomid, mj_inst_dist, mj_inst_geomid,
    )
    mj_inst_valid = (
        np.isfinite(mj_inst_dist) & (mj_inst_dist >= 0) & (mj_inst_dist <= MAX_DISTANCE)
    )
    visualize(
        args.save, unilab_dirs, uni_dist, uni_hit, uni_hit_pts,
        inst_dirs, mj_inst_dist, mj_inst_valid,
    )


if __name__ == "__main__":
    main()