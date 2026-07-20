"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import numpy as np

from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import cleanup_resources, EvalRealConfig

from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.utils import to_list, to_scalar

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

# LeRobot 数据集在不同版本里保存 action 的方式不同：
# - 旧版通常是一个整体向量列：action
# - v3 数据集常把动作拆成 action.left_arm、action.right_arm、action.left_gripper 等多列
# replay_robot 的控制逻辑仍然需要一个拼好的动作向量，因此这里声明各类末端执行器的拼接顺序。
ACTION_COLUMN_ORDER = {
    "dex1": ["action.left_gripper", "action.right_gripper"],
    "dex3": ["action.left_hand", "action.right_hand"],
    "inspire1": ["action.left_hand", "action.right_hand"],
    "brainco": ["action.left_hand", "action.right_hand"],
}


def _to_numpy_1d(value):
    """把 torch/list/numpy 类型统一转成一维 numpy 向量，便于后面拼接和切片。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _concat_row_columns(row, keys):
    """按给定 key 顺序从一帧数据里取列，并拼成 replay 控制器期待的动作/状态向量。"""
    return np.concatenate([_to_numpy_1d(row[key]) for key in keys])


def _resolve_action_keys(dataset, ee):
    """根据数据集列名选择 action 读取方式，优先兼容旧版整体 action 列。"""
    columns = set(dataset.hf_dataset.column_names)
    if "action" in columns:
        # 旧版格式：action 本身已经是完整控制向量，不需要再拆列拼接。
        return ["action"]

    # 新版展开格式：双臂固定在前面，末端执行器动作接在 arm_dof 后面。
    # 这必须和下面 action_np[:arm_dof]、后续 ee_dof 切片保持一致。
    keys = ["action.left_arm", "action.right_arm"]
    if ee:
        keys.extend(ACTION_COLUMN_ORDER.get(ee.lower(), []))

    missing = [key for key in keys if key not in columns]
    if missing:
        # 这里主动抛出可读错误，避免后面 KeyError 或动作维度错位导致机器人收到错误指令。
        action_columns = [key for key in dataset.hf_dataset.column_names if key.startswith("action.")]
        raise ValueError(
            f"Missing replay action columns {missing}. Available action columns: {action_columns}"
        )
    return keys


def _get_action(row, action_keys):
    """读取单帧动作；旧版直接取 action，新版按 action_keys 拼接。"""
    if action_keys == ["action"]:
        return _to_numpy_1d(row["action"])
    return _concat_row_columns(row, action_keys)


def _get_initial_arm_pose(step):
    """从数据集第一帧取机器人初始双臂姿态，兼容旧版 observation.state 和新版拆分列。"""
    if "observation.state" in step:
        # 旧版 observation.state 里通常包含 arm + ee 等状态，这里只取双臂 14 维用于初始化。
        return _to_numpy_1d(step["observation.state"])[:14]
    return _concat_row_columns(step, ["observation.state.left_arm", "observation.state.right_arm"])


@parser.wrap()
def replay_main(cfg: EvalRealConfig):
    """按数据集记录的动作逐帧回放到机器人，可选同步可视化。"""
    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        # Rerun 只用于观察 replay 过程，不参与控制闭环。
        rerun_logger = RerunLogger()

    # visualization=false 时不启动图像客户端，避免无显示/无相机环境下多余依赖。
    image_info = setup_image_client(cfg) if cfg.visualization else None
    robot_interface = setup_robot_interface(cfg)

    """The main control and evaluation loop."""
    # 机器人接口里包含 arm 控制器、IK 解算器、末端执行器共享内存和各自自由度。
    # 后续动作向量会按 arm_dof/ee_dof 切片，因此这里的自由度定义是 replay 的核心边界。
    arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
        robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
    )
    if cfg.visualization:
        # 可视化需要图像共享内存和相机形状信息，用于拼出当前观测并写入 Rerun。
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

    logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")

    # 只加载用户指定的 episode；replay 的 idx 会在这个子数据集范围内逐帧推进。
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root, episodes=[cfg.episodes])
    action_keys = _resolve_action_keys(dataset, cfg.ee)
    # 只保留 replay 需要的 action 列，减少每帧读取时的数据转换开销。
    actions = dataset.hf_dataset.select_columns(action_keys)

    # 用 episode 第一帧的双臂姿态作为初始化目标，让机器人先回到数据采集时的起始姿态。
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    step = dataset[from_idx]
    init_left_arm_pose = _get_initial_arm_pose(step)

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == "s":
        # 收到人工确认后才下发初始姿态，避免脚本启动时机器人立刻运动。
        logger_mp.info("Initializing robot to starting pose...")
        tau = arm_ik.solve_tau(init_left_arm_pose)
        arm_ctrl.ctrl_dual_arm(init_left_arm_pose, tau)
        time.sleep(1)
        for idx in range(dataset.num_frames):
            # 用 perf_counter 统计本轮耗时，循环尾部会补 sleep 来维持 cfg.frequency。
            loop_start_time = time.perf_counter()

            left_ee_state = right_ee_state = np.array([])
            action_np = _get_action(actions[idx], action_keys)

            # 动作向量约定：前 arm_dof 维是双臂关节目标，后面依次是左右末端执行器目标。
            arm_action = action_np[:arm_dof]
            tau = arm_ik.solve_tau(arm_action)
            arm_ctrl.ctrl_dual_arm(arm_action, tau)
            logger_mp.info(f"arm_action {arm_action}, tau {tau}")

            if cfg.ee:
                # 末端执行器按左右手各 ee_dof 维切片；dex1 是左右各 1 维，dex3 是左右各 7 维。
                ee_action_start_idx = arm_dof
                left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                # 先读取当前末端状态，主要供可视化记录 state 使用。
                with ee_shared_mem["lock"]:
                    full_state = np.array(ee_shared_mem["state"][:])
                    left_ee_state = full_state[:ee_dof]
                    right_ee_state = full_state[ee_dof:]

                # 不同末端控制器的共享内存类型不同：
                # 多自由度手是 Array；单自由度夹爪是 Value，因此写入方式要分支处理。
                if isinstance(ee_shared_mem["left"], SynchronizedArray):
                    ee_shared_mem["left"][:] = to_list(left_ee_action)
                    ee_shared_mem["right"][:] = to_list(right_ee_action)
                elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                    ee_shared_mem["left"].value = to_scalar(left_ee_action)
                    ee_shared_mem["right"].value = to_scalar(right_ee_action)

            if cfg.visualization:
                # replay 不用图像参与控制，但可视化时会采集当前图像和实际关节状态用于对照。
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                state = np.concatenate((current_arm_q, left_ee_state, right_ee_state))

                visualization_data(idx, observation, state, action_np, rerun_logger)

            # 频率控制：扣掉本轮计算/通信耗时，只 sleep 剩余时间；若超时则不 sleep。
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    if image_info is not None:
        # 只在启动过图像资源时清理，避免 visualization=false 时访问未初始化对象。
        cleanup_resources(image_info)


if __name__ == "__main__":
    replay_main()
