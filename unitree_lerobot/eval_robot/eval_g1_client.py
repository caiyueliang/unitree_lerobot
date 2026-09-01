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
from unitree_lerobot.eval_robot.robot_control.robot_arm import (
    INIT_STATE_PATH,
    _load_initial_arm_q_target,
    initialize_robot_to_starting_pose,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig, to_list, to_scalar

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

OBS_STATE_PATH = Path("./obs_state.json")


def _load_initial_arm_pose(arm_dof: int, path: Path = OBS_STATE_PATH) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create it with an 'observation.state' list.")
    with path.open("r", encoding="utf-8") as file:
        obs_state_data = json.load(file)
    if "observation.state" not in obs_state_data:
        raise ValueError(f"{path} must contain 'observation.state'.")
    init_arm_pose = np.asarray(obs_state_data["observation.state"], dtype=np.float32).reshape(-1)
    if init_arm_pose.shape[0] != arm_dof:
        raise ValueError(
            f"{path} observation.state must contain {arm_dof} values, got {init_arm_pose.shape[0]}."
        )
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


def _metadata_observation_keys(metadata: dict) -> list[str]:
    data_keys = metadata.get("data_keys")
    if data_keys:
        return list(data_keys)
    return list(metadata.get("obs_delta_indices", {}))


def eval_policy_client(cfg: EvalRealConfig, remote_policy: RemotePolicy):
    """Collect live observations, request actions from the remote VLA service, and execute them."""
    logger_mp.info(f"Arguments: {cfg}")
    metadata = remote_policy.metadata
    control_space = metadata.get("control_space", "")
    expected_token = os.environ.get("UNIBOT_SUBMISSION_TOKEN")
    observation_keys = _metadata_observation_keys(metadata)

    logger_mp.info(
        "Remote policy metadata: control_space=%s, observation_keys=%s, obs_delta_indices=%s, obs_chunk_size=%s, action_chunk_size=%s",
        control_space,
        observation_keys,
        metadata.get("obs_delta_indices"),
        metadata.get("obs_chunk_size"),
        metadata.get("action_chunk_size"),
    )

    if cfg.visualization:
        rerun_logger = RerunLogger()

    idx = 0
    image_client = None
    arm_ctrl = None
    arm_ik = None
    arm_dof = None
    init_arm_pose = None
    should_restore_robot = True
    try:
        image_client, image_config = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )

        policy_task = cfg.task.strip()
        if not policy_task:
            raise ValueError("--task is required when running eval_g1_client without a dataset.")
        logger_mp.info("Using remote policy task: %s", policy_task)

        init_arm_pose = _load_initial_arm_pose(arm_dof, OBS_STATE_PATH)
        logger_mp.info("Loaded initial arm pose from %s", OBS_STATE_PATH)

        reset_reply = remote_policy.reset()
        logger_mp.info("remote_policy.reset() -> %s", reset_reply)

        # # 输入's'机器人才会开始运动
        # user_input = input("Enter 's' to initialize the robot and start the remote evaluation: ")
        # print(f"user_input: {user_input}")
        # if user_input.lower() != "s":
        #     logger_mp.info("User did not start evaluation.")
        #     return

        # 机器人移动到初始位置
        initialize_robot_to_starting_pose(arm_ctrl, arm_ik, init_arm_pose, cfg.send_real_robot, wait_s=1.0)

        logger_mp.info(f"Starting remote evaluation loop at {cfg.frequency} Hz.")
        while idx < cfg.max_steps:
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

            for chunk_idx in range(robot_action.action_sequence.shape[0]):
                frame_start_time = time.perf_counter()
                arm_action = robot_action.arm[chunk_idx]
                left_ee_action = robot_action.left_ee[chunk_idx]
                right_ee_action = robot_action.right_ee[chunk_idx]

                if idx % 30 == 0:
                    max_arm_delta = float(np.max(np.abs(arm_action - current_arm_q)))
                    logger_mp.info(
                        "Remote action frame %d chunk[%d/%d]: max_arm_delta=%.5f, current_arm_q=%s",
                        idx,
                        chunk_idx + 1,
                        robot_action.action_sequence.shape[0],
                        max_arm_delta,
                        np.array2string(current_arm_q, precision=4, suppress_small=True),
                    )
                    logger_mp.info(
                        "action sequence shape=%s action_sequence=\n%s",
                        robot_action.action_sequence.shape,
                        robot_action.action_sequence,
                    )

                if cfg.send_real_robot:
                    tau = arm_ik.solve_tau(arm_action)
                    arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee and cfg.send_real_robot:
                    if left_ee_action.shape[0] != ee_dof or right_ee_action.shape[0] != ee_dof:
                        raise ValueError(
                            f"Remote gripper action dim mismatch: left={left_ee_action.shape}, "
                            f"right={right_ee_action.shape}, ee_dof={ee_dof}"
                        )
                    _write_ee_action(ee_shared_mem, left_ee_action, right_ee_action)

                if cfg.visualization:
                    state_tensor = np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                    action_np = robot_action.action_sequence[chunk_idx]
                    visualization_data(idx, observation, state_tensor, action_np, rerun_logger)

                idx += 1
                if idx >= cfg.max_steps:
                    break
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - frame_start_time)))
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if should_restore_robot and arm_ctrl is not None and arm_ik is not None and init_arm_pose is not None:
            try:
                logger_mp.info("Restoring robot arm to initial poses...")
                initialize_robot_to_starting_pose(arm_ctrl, arm_ik, init_arm_pose, cfg.send_real_robot, wait_s=1.0)
                init_state_pose = _load_initial_arm_q_target(arm_dof, INIT_STATE_PATH)
                initialize_robot_to_starting_pose(arm_ctrl, arm_ik, init_state_pose, cfg.send_real_robot, wait_s=1.0)
            except Exception as restore_error:
                logger_mp.info(f"An error occurred while restoring robot arm: {restore_error}")
        if image_client is not None:
            image_client.close()


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    """Connect to the VLA service, then run real-robot remote evaluation."""
    logging.info(pformat(asdict(cfg)))

    logging.info("Connecting to remote policy server: %s", cfg.policy_server_uri)
    remote_policy = RemotePolicy(host=cfg.policy_server_uri)
    eval_policy_client(cfg, remote_policy)

    logging.info("End of remote eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
