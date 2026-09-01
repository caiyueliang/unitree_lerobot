import numpy as np
import pytest
import torch

from unitree_lerobot.eval_robot.eval_g1_client_utils import (
    build_remote_observation,
    remote_action_to_robot_action,
)


def test_build_remote_observation_uses_latest_robot_state_and_repeats_chunk():
    image = torch.arange(2 * 3 * 3, dtype=torch.uint8).reshape(2, 3, 3)
    observation = {
        "observation.images.cam_left_high": image,
        "observation.images.cam_left_wrist": image + 1,
    }
    current_arm_q = np.arange(14, dtype=np.float32)
    left_gripper = np.array([0.25], dtype=np.float32)
    right_gripper = np.array([0.75], dtype=np.float32)
    metadata = {
        "obs_chunk_size": 2,
        "data_keys": [
            "observation.language",
            "observation.images.cam_left_high",
            "observation.images.cam_left_wrist",
            "observation.state.left_arm",
            "observation.state.right_arm",
            "observation.state.left_gripper",
            "observation.state.right_gripper",
        ],
    }

    remote_obs = build_remote_observation(
        observation,
        current_arm_q,
        left_gripper,
        right_gripper,
        task="place tubes",
        metadata=metadata,
    )

    assert remote_obs["observation.language"] == "place tubes"
    np.testing.assert_array_equal(
        remote_obs["observation.images.cam_left_high"],
        np.repeat(image.numpy()[None, ...], 2, axis=0),
    )
    np.testing.assert_array_equal(
        remote_obs["observation.state.left_arm"],
        np.repeat(np.arange(7, dtype=np.float32)[None, ...], 2, axis=0),
    )
    np.testing.assert_array_equal(
        remote_obs["observation.state.right_arm"],
        np.repeat(np.arange(7, 14, dtype=np.float32)[None, ...], 2, axis=0),
    )
    assert remote_obs["observation.state.left_gripper"].dtype == np.float32
    assert remote_obs["observation.state.left_gripper"].shape == (2, 1)


def test_build_remote_observation_uses_obs_delta_indices_when_data_keys_missing():
    image = torch.arange(2 * 3 * 3, dtype=torch.uint8).reshape(2, 3, 3)
    observation = {
        "observation.images.cam_left_high": image,
        "observation.images.cam_left_wrist": image + 1,
        "observation.images.cam_right_wrist": image + 2,
    }
    current_arm_q = np.arange(14, dtype=np.float32)
    metadata = {
        "obs_delta_indices": {
            "observation.language": [0],
            "observation.images.cam_left_high": [0],
            "observation.images.cam_left_wrist": [0],
            "observation.images.cam_right_wrist": [0],
            "observation.state.left_arm": [0],
            "observation.state.right_arm": [0],
            "observation.state.left_gripper": [0],
            "observation.state.right_gripper": [0],
        },
    }

    remote_obs = build_remote_observation(
        observation,
        current_arm_q,
        np.array([0.25], dtype=np.float32),
        np.array([0.75], dtype=np.float32),
        task="place tubes",
        metadata=metadata,
    )

    assert set(remote_obs) == set(metadata["obs_delta_indices"])
    np.testing.assert_array_equal(remote_obs["observation.images.cam_left_high"], image.numpy()[None, ...])
    np.testing.assert_array_equal(remote_obs["observation.images.cam_left_wrist"], (image + 1).numpy()[None, ...])
    np.testing.assert_array_equal(remote_obs["observation.images.cam_right_wrist"], (image + 2).numpy()[None, ...])


def test_remote_action_to_robot_action_uses_first_joint_chunk_frame():
    action = {
        "meta.token": "123456",
        "action.left_arm": np.array([[1, 2, 3, 4, 5, 6, 7], [10, 20, 30, 40, 50, 60, 70]], dtype=np.float32),
        "action.right_arm": np.array([[8, 9, 10, 11, 12, 13, 14]], dtype=np.float32),
        "action.left_gripper": np.array([[0.2], [0.4]], dtype=np.float32),
        "action.right_gripper": np.array([[0.8]], dtype=np.float32),
        "action.pivot": np.zeros((1, 7), dtype=np.float32),
    }

    robot_action = remote_action_to_robot_action(action, control_space="joint", expected_token="123456")

    np.testing.assert_array_equal(robot_action.arm, np.arange(1, 15, dtype=np.float32))
    np.testing.assert_array_equal(robot_action.left_ee, np.array([0.2], dtype=np.float32))
    np.testing.assert_array_equal(robot_action.right_ee, np.array([0.8], dtype=np.float32))


def test_remote_action_to_robot_action_rejects_ee_control_space():
    with pytest.raises(NotImplementedError, match="control_space='ee'"):
        remote_action_to_robot_action({"meta.token": "123456"}, control_space="ee", expected_token="123456")
