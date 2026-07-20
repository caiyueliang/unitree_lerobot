"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    """在线运行策略模型：采集机器人观测、推理动作，并按配置选择是否下发到真实机器人。"""
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        # Rerun 只用于记录和查看运行过程，不改变策略输入或机器人控制输出。
        rerun_logger = RerunLogger()

    # 每次评估前重置策略和处理器内部缓存。
    # 对 diffusion/ACT 等带动作队列或时序窗口的策略尤其重要，否则可能混入上一次运行的状态。
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    image_client = None
    image_config = None
    try:
        # --- 初始化阶段 ---
        # 图像客户端连接机器人端相机服务；机器人接口初始化双臂、IK、末端执行器共享内存等控制资源。
        image_client, image_config = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # 解包常用接口。arm_dof/ee_dof 决定 action_np 的切片边界，必须和训练数据/策略输出一致。
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )

        # 使用数据集第一帧的 observation.state 作为初始双臂姿态参考。
        # 这样真实机器人开始推理前，会先移动到和数据采集起点相近的位置。
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        if user_input.lower() == "s":
            # 人工输入 s 后才允许初始化运动，防止脚本启动瞬间机器人直接执行动作。
            logger_mp.info("Initializing robot to starting pose...")
            if cfg.send_real_robot:
                # 真实下发时先通过 IK 求解 tau，再控制双臂到数据集起始姿态。
                tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
                robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
                time.sleep(1.0)
            else:
                logger_mp.warning(
                    "send_real_robot=false：跳过机器人初始姿态运动。"
                )
            # --- 主循环阶段 ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                # 用 perf_counter 统计每一帧总耗时，循环尾部据此补 sleep 保持固定频率。
                loop_start_time = time.perf_counter()
                # 1. 采集当前观测：相机图像 + 当前双臂关节状态。
                observation, current_arm_q = process_images_and_observations(
                    image_client,
                    image_config,
                    arm_ctrl,
                )
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    # 末端执行器控制进程通过共享内存暴露当前状态。
                    # state 数组约定为 [left_ee_state, right_ee_state]。
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                # 策略期望 observation.state 是一个连续向量：
                # 双臂当前关节 + 左末端状态 + 右末端状态。
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                observation["observation.state"] = state_tensor
                # 2. 策略推理动作。
                # predict_action 内部会走 preprocessor -> policy -> postprocessor，并可按 cfg.use_dataset 使用数据集动作。
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                    robot_type=None,
                )
                action_np = action.cpu().numpy()
                # 3. 执行动作。
                # 策略输出向量约定：前 arm_dof 维是双臂动作，后面按左右末端执行器各 ee_dof 维排列。
                arm_action = action_np[:arm_dof]

                if cfg.send_real_robot:
                    # send_real_robot 是真实机器人下发总开关；false 时只推理和可视化，不控制机器人。
                    tau = arm_ik.solve_tau(arm_action)
                    arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee and cfg.send_real_robot:
                    # 只有启用末端执行器且允许真实下发时，才把末端动作写入共享内存。
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    # 多自由度手使用 Array；单自由度夹爪使用 Value。
                    # 写入共享内存后，末端执行器控制线程会从这里读取目标值并下发。
                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.visualization:
                    # 可视化记录当前观测、拼接后的状态向量和策略动作，便于离线排查策略/机器人响应。
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # 频率控制：扣除本轮采集、推理、通信耗时，只 sleep 剩余时间；超时则立即进入下一帧。
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except Exception as e:
        # 这里保留原有吞异常行为：记录错误后进入 finally 做资源释放。
        # 如果需要调试完整堆栈，可以临时改成 logger_mp.exception。
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if image_client is not None:
            # 图像客户端是外部连接资源，主循环异常退出时也要关闭。
            image_client.close()


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    """解析配置、加载数据集与策略权重，然后进入真实机器人在线评估。"""
    logging.info(pformat(asdict(cfg)))

    # 检查策略配置里的 device 是否可用，例如 cuda/cpu；不可用时会按 LeRobot 逻辑安全降级或报错。
    device = get_safe_torch_device(cfg.policy.device, log=True)

    # cuDNN benchmark 适合固定输入尺寸场景；allow_tf32 可以提高 Ampere 及以上 GPU 的 matmul 性能。
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    # root 为空时走 LeRobot/Hugging Face 默认缓存；指定 root 时从本地数据集目录读取。
    dataset_kwargs = {"repo_id": cfg.repo_id}
    if cfg.root:
        dataset_kwargs["root"] = cfg.root

    dataset = LeRobotDataset(**dataset_kwargs)

    # 根据 policy 配置和数据集元信息构建策略模型，并切换到 eval 模式关闭训练行为。
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    # 构建推理前后的处理流水线：
    # preprocessor 负责重命名、搬到 device、归一化等；postprocessor 负责反归一化和动作格式处理。
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    # 推理阶段不需要梯度；use_amp=true 时开启自动混合精度以降低显存和提升速度。
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
