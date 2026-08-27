"""Real-robot evaluation client for a remote UniBot VLA policy service."""

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import numpy as np
from multiprocessing.sharedctypes import SynchronizedArray

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging
from policy.web_policy import RemotePolicy

from unitree_lerobot.eval_robot.eval_g1_client_utils import (
    build_remote_observation,
    remote_action_to_robot_action,
)
from unitree_lerobot.eval_robot.make_robot import (
    process_images_and_observations,
    setup_image_client,
    setup_robot_interface,
)
from unitree_lerobot.eval_robot.task_selection import list_unique_tasks
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig, to_list, to_scalar

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

OBS_STATE_PATH = Path("./obs_state.json")
STANDARD_ARM_JOINT_INDICES = (0, 1, 2, 3, 5, 6, 4)


def _to_numpy_1d(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _standardize_split_arm(value):
    arm = _to_numpy_1d(value)
    if arm.shape[0] != len(STANDARD_ARM_JOINT_INDICES):
        raise ValueError(f"Expected split arm state with 7 values, got shape {arm.shape}")
    return arm[list(STANDARD_ARM_JOINT_INDICES)]


def _get_initial_arm_pose(step, arm_dof):
    """Read the first-frame dual-arm pose from old vector datasets or v3 split datasets."""
    if "observation.state" in step:
        return _to_numpy_1d(step["observation.state"])[:arm_dof]

    required_keys = ("observation.state.left_arm", "observation.state.right_arm")
    missing_keys = [key for key in required_keys if key not in step]
    if missing_keys:
        available_state_keys = [key for key in step if key.startswith("observation.state")]
        raise KeyError(
            f"Missing initial arm state keys {missing_keys}. Available observation state keys: {available_state_keys}"
        )

    init_arm_pose = np.concatenate(
        (
            _standardize_split_arm(step["observation.state.left_arm"]),
            _standardize_split_arm(step["observation.state.right_arm"]),
        )
    )
    if init_arm_pose.shape[0] != arm_dof:
        raise ValueError(f"Expected initial arm pose with {arm_dof} values, got shape {init_arm_pose.shape}")
    return init_arm_pose


def _read_ee_state(cfg: EvalRealConfig, ee_shared_mem, ee_dof: int):
    if not cfg.ee:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), None
    with ee_shared_mem["lock"]:
        full_state = np.array(ee_shared_mem["state"][:], dtype=np.float32)
        left_ee_state = full_state[:ee_dof]
        right_ee_state = full_state[ee_dof:]
    return left_ee_state, right_ee_state, full_state


def _write_ee_action(ee_shared_mem, left_ee_action: np.ndarray, right_ee_action: np.ndarray):
    if isinstance(ee_shared_mem["left"], SynchronizedArray):
        ee_shared_mem["left"][:] = to_list(left_ee_action)
        ee_shared_mem["right"][:] = to_list(right_ee_action)
    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
        ee_shared_mem["left"].value = to_scalar(left_ee_action)
        ee_shared_mem["right"].value = to_scalar(right_ee_action)


def eval_policy_client(cfg: EvalRealConfig, dataset: LeRobotDataset, remote_policy: RemotePolicy):
    """Collect live observations, request actions from the remote VLA service, and execute them."""
    logger_mp.info(f"Arguments: {cfg}")
    metadata = remote_policy.metadata
    control_space = metadata.get("control_space", "")
    expected_token = os.environ.get("UNIBOT_SUBMISSION_TOKEN")

    logger_mp.info(
        "Remote policy metadata: control_space=%s, data_keys=%s, obs_chunk_size=%s, action_chunk_size=%s",
        control_space,
        metadata.get("data_keys"),
        metadata.get("obs_chunk_size"),
        metadata.get("action_chunk_size"),
    )

    if cfg.visualization:
        rerun_logger = RerunLogger()

    image_client = None
    try:
        image_client, image_config = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )

        # 优先使用本地缓存的 observation.state；没有缓存时才从数据集第一帧读取并保存。
        # 这样真实机器人开始推理前，会先移动到和数据采集起点相近的位置。
        need_dataset_step = not OBS_STATE_PATH.exists() or not cfg.task.strip()
        step = None
        if need_dataset_step:
            from_idx = dataset.meta.episodes["dataset_from_index"][0]
            step = dataset[from_idx]

        policy_task = cfg.task.strip() or step["task"]
        logger_mp.info("Using remote policy task: %s", policy_task)

        if OBS_STATE_PATH.exists():
            with OBS_STATE_PATH.open("r", encoding="utf-8") as file:
                obs_state_data = json.load(file)
            if "observation.state" not in obs_state_data:
                raise ValueError(f"{OBS_STATE_PATH} must contain 'observation.state'.")
            init_arm_pose = np.asarray(obs_state_data["observation.state"], dtype=np.float32).reshape(-1)
            if init_arm_pose.shape[0] != arm_dof:
                raise ValueError(
                    f"{OBS_STATE_PATH} observation.state must contain {arm_dof} values, "
                    f"got {init_arm_pose.shape[0]}."
                )
            logger_mp.info("Loaded initial arm pose from %s", OBS_STATE_PATH)
        else:
            init_arm_pose = _get_initial_arm_pose(step, arm_dof)
            OBS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with OBS_STATE_PATH.open("w", encoding="utf-8") as file:
                json.dump({"observation.state": init_arm_pose.tolist()}, file, indent=2)
            logger_mp.info("Saved initial arm pose to %s", OBS_STATE_PATH)

        reset_reply = remote_policy.reset()
        logger_mp.info("remote_policy.reset() -> %s", reset_reply)

        # # 机器人移动到初始位置
        # logger_mp.info("Initializing robot to starting pose...")
        # if cfg.send_real_robot:
        #     tau = arm_ik.solve_tau(init_arm_pose)
        #     arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
        #     time.sleep(1.0)
        # else:
        #     logger_mp.warning("send_real_robot=false：跳过机器人初始姿态运动。")

        # 输入's'机器人才会开始运动
        user_input = input("Enter 's' to initialize the robot and start the remote evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        if user_input.lower() != "s":
            logger_mp.info("User did not start evaluation.")
            return

        # 机器人移动到初始位置
        logger_mp.info("Initializing robot to starting pose...")
        if cfg.send_real_robot:
            tau = arm_ik.solve_tau(init_arm_pose)
            arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)
        else:
            logger_mp.warning("send_real_robot=false：跳过机器人初始姿态运动。")

        logger_mp.info(f"Starting remote evaluation loop at {cfg.frequency} Hz.")
        while True:
            loop_start_time = time.perf_counter()
            observation, current_arm_q = process_images_and_observations(image_client, image_config, arm_ctrl)
            if current_arm_q is None:
                raise RuntimeError("Failed to read current arm state.")

            left_ee_state, right_ee_state, _ = _read_ee_state(cfg, ee_shared_mem, ee_dof)
            remote_observation = build_remote_observation(
                observation,
                current_arm_q,
                left_ee_state,
                right_ee_state,
                task=policy_task,
                metadata=metadata,
            )
            # 调用推理服务，获取下一步的动作
            remote_action = remote_policy.get_action(remote_observation)
            robot_action = remote_action_to_robot_action(
                remote_action,
                control_space=control_space,
                expected_token=expected_token,
            )

            arm_action = robot_action.arm
            if idx % 30 == 0:
                max_arm_delta = float(np.max(np.abs(arm_action - current_arm_q)))
                logger_mp.info(
                    "Remote action frame %d: max_arm_delta=%.5f, current_arm_q=%s, arm_action=%s",
                    idx,
                    max_arm_delta,
                    np.array2string(current_arm_q, precision=4, suppress_small=True),
                    np.array2string(arm_action, precision=4, suppress_small=True),
                )

            if cfg.send_real_robot:
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

            if cfg.ee and cfg.send_real_robot:
                if robot_action.left_ee.shape[0] != ee_dof or robot_action.right_ee.shape[0] != ee_dof:
                    raise ValueError(
                        f"Remote gripper action dim mismatch: left={robot_action.left_ee.shape}, "
                        f"right={robot_action.right_ee.shape}, ee_dof={ee_dof}"
                    )
                _write_ee_action(ee_shared_mem, robot_action.left_ee, robot_action.right_ee)

            if cfg.visualization:
                state_tensor = np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                action_np = np.concatenate((arm_action, robot_action.left_ee, robot_action.right_ee), axis=0)
                visualization_data(idx, observation, state_tensor, action_np, rerun_logger)

            idx += 1
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if image_client is not None:
            image_client.close()


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    """Load dataset metadata, connect to the VLA service, then run real-robot remote evaluation."""
    logging.info(pformat(asdict(cfg)))

    dataset_kwargs = {"repo_id": cfg.repo_id}
    if cfg.root:
        dataset_kwargs["root"] = cfg.root
    dataset = LeRobotDataset(**dataset_kwargs)
    unique_tasks = list_unique_tasks(dataset.meta.episodes)
    logging.warning('Unique dataset.meta.episodes[*]["tasks"] (%d): %s', len(unique_tasks), unique_tasks)

    logging.info("Connecting to remote policy server: %s", cfg.policy_server_uri)
    remote_policy = RemotePolicy(host=cfg.policy_server_uri)
    eval_policy_client(cfg, dataset, remote_policy)

    logging.info("End of remote eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
