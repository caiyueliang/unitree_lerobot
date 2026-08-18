from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RobotAction:
    arm: np.ndarray
    left_ee: np.ndarray
    right_ee: np.ndarray


def _to_numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _repeat_chunk(frame: Any, chunk_size: int, dtype: np.dtype | type | None = None) -> np.ndarray:
    array = _to_numpy(frame, dtype=dtype)
    return np.repeat(array[None, ...], chunk_size, axis=0)


def _state_chunk(frame: Any, dim: int, chunk_size: int) -> np.ndarray:
    array = _to_numpy(frame, np.float32).reshape(-1)
    if array.shape[0] != dim:
        raise ValueError(f"Expected state with {dim} values, got shape {array.shape}")
    return _repeat_chunk(array, chunk_size, np.float32)


def build_remote_observation(
    observation: dict[str, Any],
    current_arm_q: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    task: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    obs_chunk_size = int(metadata.get("obs_chunk_size", 1))
    data_keys = list(metadata.get("data_keys", []))
    current_arm_q = _to_numpy(current_arm_q, np.float32).reshape(-1)
    if current_arm_q.shape[0] < 14:
        raise ValueError(f"Expected current_arm_q with at least 14 values, got shape {current_arm_q.shape}")

    source_values = {
        "observation.language": task,
        "observation.state.left_arm": current_arm_q[:7],
        "observation.state.right_arm": current_arm_q[7:14],
        "observation.state.left_gripper": _to_numpy(left_gripper, np.float32).reshape(-1),
        "observation.state.right_gripper": _to_numpy(right_gripper, np.float32).reshape(-1),
        "observation.state.lower_body": np.zeros((15,), dtype=np.float32),
        "observation.state.left_ee_pose_gripper_base": np.zeros((6,), dtype=np.float32),
        "observation.state.right_ee_pose_gripper_base": np.zeros((6,), dtype=np.float32),
    }

    remote_obs: dict[str, Any] = {}
    for key in data_keys:
        if key == "observation.language":
            remote_obs[key] = source_values[key]
        elif key.startswith("observation.images."):
            if key not in observation or observation[key] is None:
                raise KeyError(f"Missing image observation {key!r}")
            remote_obs[key] = _repeat_chunk(observation[key], obs_chunk_size, np.uint8)
        elif key in ("observation.state.left_arm", "observation.state.right_arm"):
            remote_obs[key] = _state_chunk(source_values[key], 7, obs_chunk_size)
        elif key in ("observation.state.left_gripper", "observation.state.right_gripper"):
            remote_obs[key] = _state_chunk(source_values[key], 1, obs_chunk_size)
        elif key == "observation.state.lower_body":
            remote_obs[key] = _state_chunk(source_values[key], 15, obs_chunk_size)
        elif key in (
            "observation.state.left_ee_pose_gripper_base",
            "observation.state.right_ee_pose_gripper_base",
        ):
            remote_obs[key] = _state_chunk(source_values[key], 6, obs_chunk_size)
        else:
            raise KeyError(f"Unsupported remote observation key {key!r}")
    return remote_obs


def _first_action_frame(action: dict[str, Any], key: str, dim: int) -> np.ndarray:
    if key not in action:
        raise KeyError(f"Missing remote action key {key!r}")
    array = _to_numpy(action[key], np.float32)
    if array.shape == (dim,):
        return array
    if array.ndim == 2 and array.shape[1] == dim and array.shape[0] > 0:
        return array[0]
    raise ValueError(f"Remote action {key!r} shape {array.shape} cannot be used as (*, {dim})")


def remote_action_to_robot_action(
    action: dict[str, Any],
    control_space: str,
    expected_token: str | None = None,
) -> RobotAction:
    token = action.get("meta.token")
    if expected_token is not None and token != expected_token:
        raise ValueError("Remote action meta.token does not match expected token")
    if control_space == "ee":
        raise NotImplementedError("Remote control_space='ee' is not supported by eval_g1_client joint control")
    if control_space != "joint":
        raise ValueError(f"Unsupported remote control_space {control_space!r}")

    left_arm = _first_action_frame(action, "action.left_arm", 7)
    right_arm = _first_action_frame(action, "action.right_arm", 7)
    return RobotAction(
        arm=np.concatenate((left_arm, right_arm), axis=0).astype(np.float32, copy=False),
        left_ee=_first_action_frame(action, "action.left_gripper", 1).astype(np.float32, copy=False),
        right_ee=_first_action_frame(action, "action.right_gripper", 1).astype(np.float32, copy=False),
    )
