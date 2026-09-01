import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from unitree_lerobot.eval_robot import eval_g1_client
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig


class FakeRemotePolicy:
    metadata = {
        "control_space": "joint",
        "data_keys": [
            "observation.language",
            "observation.state.left_arm",
            "observation.state.right_arm",
        ],
        "obs_chunk_size": 1,
        "action_chunk_size": 1,
    }

    def __init__(self):
        self.reset_count = 0
        self.action_count = 0

    def reset(self):
        self.reset_count += 1
        return {"ok": True}

    def get_action(self, observation):
        self.action_count += 1
        return {
            "action.left_arm": np.zeros(7, dtype=np.float32),
            "action.right_arm": np.zeros(7, dtype=np.float32),
            "action.left_gripper": np.zeros(1, dtype=np.float32),
            "action.right_gripper": np.zeros(1, dtype=np.float32),
        }


class FakeImageClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def fake_cfg(task="cached task", max_steps=0):
    return SimpleNamespace(visualization=False, task=task, send_real_robot=False, frequency=30.0, ee="", max_steps=max_steps)


def fake_robot_interface():
    arm_ik = Mock()
    arm_ik.solve_tau.return_value = np.zeros(14, dtype=np.float32)
    arm_ctrl = Mock()
    return {
        "arm_ctrl": arm_ctrl,
        "arm_ik": arm_ik,
        "ee_shared_mem": {},
        "arm_dof": 14,
        "ee_dof": 0,
    }


class ObsStateCacheTest(unittest.TestCase):
    def run_client_until_prompt(self, cfg, remote_policy, obs_state_path):
        image_client = FakeImageClient()
        with patch.object(eval_g1_client, "OBS_STATE_PATH", obs_state_path), patch.object(
            eval_g1_client, "setup_image_client", return_value=(image_client, {})
        ), patch.object(eval_g1_client, "setup_robot_interface", return_value=fake_robot_interface()), patch(
            "builtins.input", return_value=""
        ):
            eval_g1_client.eval_policy_client(cfg, remote_policy)
        self.assertTrue(image_client.closed)

    def test_load_initial_arm_pose_reads_local_obs_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            cached_state = [float(value) for value in range(100, 114)]
            obs_state_path.write_text(json.dumps({"observation.state": cached_state}), encoding="utf-8")

            init_arm_pose = eval_g1_client._load_initial_arm_pose(14, obs_state_path)

            np.testing.assert_array_equal(init_arm_pose, np.asarray(cached_state, dtype=np.float32))

    def test_existing_obs_state_cache_is_used_without_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            cached_state = [float(value) for value in range(100, 114)]
            obs_state_path.write_text(json.dumps({"observation.state": cached_state}), encoding="utf-8")
            remote_policy = FakeRemotePolicy()

            self.run_client_until_prompt(fake_cfg(task="explicit task"), remote_policy, obs_state_path)

            self.assertEqual(remote_policy.reset_count, 1)
            self.assertEqual(json.loads(obs_state_path.read_text(encoding="utf-8"))["observation.state"], cached_state)

    def test_missing_obs_state_cache_stops_before_remote_policy_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            remote_policy = FakeRemotePolicy()

            self.run_client_until_prompt(fake_cfg(task="explicit task"), remote_policy, obs_state_path)

            self.assertEqual(remote_policy.reset_count, 0)

    def test_invalid_obs_state_cache_stops_before_remote_policy_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            obs_state_path.write_text(json.dumps({"observation.state": [1.0, 2.0]}), encoding="utf-8")
            remote_policy = FakeRemotePolicy()

            self.run_client_until_prompt(fake_cfg(task="explicit task"), remote_policy, obs_state_path)

            self.assertEqual(remote_policy.reset_count, 0)

    def test_blank_task_stops_before_remote_policy_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            obs_state_path.write_text(
                json.dumps({"observation.state": [float(value) for value in range(14)]}), encoding="utf-8"
            )
            remote_policy = FakeRemotePolicy()

            self.run_client_until_prompt(fake_cfg(task=""), remote_policy, obs_state_path)

            self.assertEqual(remote_policy.reset_count, 0)

    def test_eval_main_does_not_construct_lerobot_dataset(self):
        cfg = EvalRealConfig(repo_id="", task="explicit task")
        remote_policy = FakeRemotePolicy()

        with patch.object(eval_g1_client, "LeRobotDataset", side_effect=AssertionError("dataset should not load"), create=True), patch.object(
            eval_g1_client, "RemotePolicy", return_value=remote_policy
        ), patch.object(eval_g1_client, "eval_policy_client") as eval_policy_client:
            eval_g1_client.eval_main.__wrapped__(cfg)

        eval_policy_client.assert_called_once_with(cfg, remote_policy)

    def test_eval_real_config_defaults_max_steps_to_one_minute_at_30hz(self):
        cfg = EvalRealConfig(repo_id="", task="explicit task")

        self.assertEqual(cfg.max_steps, 60 * 30)

    def test_metadata_observation_keys_use_obs_delta_indices_without_data_keys(self):
        metadata = {
            "control_space": "joint",
            "obs_delta_indices": {
                "observation.language": [0],
                "observation.images.cam_left_high": [0],
                "observation.state.left_arm": [0],
            },
            "action_chunk_size": 1,
        }

        self.assertEqual(
            eval_g1_client._metadata_observation_keys(metadata),
            [
                "observation.language",
                "observation.images.cam_left_high",
                "observation.state.left_arm",
            ],
        )

    def test_remote_eval_loop_stops_after_configured_max_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            obs_state_path.write_text(
                json.dumps({"observation.state": [float(value) for value in range(14)]}), encoding="utf-8"
            )
            remote_policy = FakeRemotePolicy()

            image_client = FakeImageClient()
            with patch.object(eval_g1_client, "OBS_STATE_PATH", obs_state_path), patch.object(
                eval_g1_client, "setup_image_client", return_value=(image_client, {})
            ), patch.object(eval_g1_client, "setup_robot_interface", return_value=fake_robot_interface()), patch(
                "builtins.input", return_value="s"
            ), patch.object(
                eval_g1_client,
                "process_images_and_observations",
                return_value=({}, np.zeros(14, dtype=np.float32)),
            ), patch.object(
                eval_g1_client.time, "sleep"
            ):
                eval_g1_client.eval_policy_client(fake_cfg(task="explicit task", max_steps=2), remote_policy)

            self.assertEqual(remote_policy.action_count, 2)
            self.assertTrue(image_client.closed)

    def test_remote_eval_loop_executes_all_remote_action_chunk_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            obs_state_path.write_text(
                json.dumps({"observation.state": [float(value) for value in range(14)]}), encoding="utf-8"
            )
            remote_policy = FakeRemotePolicy()
            remote_policy.get_action = Mock(
                return_value={
                    "action.left_arm": np.array(
                        [
                            [1, 2, 3, 4, 5, 6, 7],
                            [11, 12, 13, 14, 15, 16, 17],
                            [21, 22, 23, 24, 25, 26, 27],
                        ],
                        dtype=np.float32,
                    ),
                    "action.right_arm": np.array(
                        [
                            [8, 9, 10, 11, 12, 13, 14],
                            [18, 19, 20, 21, 22, 23, 24],
                            [28, 29, 30, 31, 32, 33, 34],
                        ],
                        dtype=np.float32,
                    ),
                    "action.left_gripper": np.zeros((3, 1), dtype=np.float32),
                    "action.right_gripper": np.zeros((3, 1), dtype=np.float32),
                }
            )
            arm_ctrl = Mock()
            arm_ik = Mock()
            arm_ik.solve_tau.return_value = np.zeros(14, dtype=np.float32)
            robot_interface = {
                "arm_ctrl": arm_ctrl,
                "arm_ik": arm_ik,
                "ee_shared_mem": {},
                "arm_dof": 14,
                "ee_dof": 0,
            }
            image_client = FakeImageClient()

            with patch.object(eval_g1_client, "OBS_STATE_PATH", obs_state_path), patch.object(
                eval_g1_client, "setup_image_client", return_value=(image_client, {})
            ), patch.object(eval_g1_client, "setup_robot_interface", return_value=robot_interface), patch.object(
                eval_g1_client,
                "process_images_and_observations",
                return_value=({}, np.zeros(14, dtype=np.float32)),
            ), patch.object(
                eval_g1_client, "initialize_robot_to_starting_pose"
            ), patch.object(
                eval_g1_client.time, "sleep"
            ), patch.object(
                eval_g1_client.logger_mp, "info"
            ) as log_info:
                cfg = fake_cfg(task="explicit task", max_steps=3)
                cfg.send_real_robot = True
                eval_g1_client.eval_policy_client(cfg, remote_policy)

            self.assertEqual(remote_policy.get_action.call_count, 1)
            actual_actions = [call_args.args[0] for call_args in arm_ctrl.ctrl_dual_arm.call_args_list]
            expected_actions = [
                np.arange(1, 15, dtype=np.float32),
                np.arange(11, 25, dtype=np.float32),
                np.arange(21, 35, dtype=np.float32),
            ]
            for actual, expected in zip(actual_actions, expected_actions, strict=True):
                np.testing.assert_array_equal(actual, expected)
            action_sequence_logs = [
                call_args
                for call_args in log_info.call_args_list
                if call_args.args and "Remote action sequence" in call_args.args[0]
            ]
            self.assertEqual(len(action_sequence_logs), 1)
            self.assertEqual(action_sequence_logs[0].args[1], (3, 16))
            np.testing.assert_array_equal(
                action_sequence_logs[0].args[2],
                np.array(
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0],
                        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 0],
                        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 0, 0],
                    ],
                    dtype=np.float32,
                ),
            )

    def test_restores_from_obs_state_to_init_state_after_max_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            init_state_path = Path(temp_dir) / "init_state.json"
            obs_pose = np.arange(14, dtype=np.float32) + 10
            init_pose = np.arange(14, dtype=np.float32)
            obs_state_path.write_text(json.dumps({"observation.state": obs_pose.tolist()}), encoding="utf-8")
            init_state_path.write_text(json.dumps({"observation.state": init_pose.tolist()}), encoding="utf-8")
            remote_policy = FakeRemotePolicy()

            image_client = FakeImageClient()
            with patch.object(eval_g1_client, "OBS_STATE_PATH", obs_state_path), patch.object(
                eval_g1_client, "INIT_STATE_PATH", init_state_path, create=True
            ), patch.object(eval_g1_client, "setup_image_client", return_value=(image_client, {})), patch.object(
                eval_g1_client, "setup_robot_interface", return_value=fake_robot_interface()
            ), patch(
                "builtins.input", return_value="s"
            ), patch.object(
                eval_g1_client,
                "process_images_and_observations",
                return_value=({}, np.zeros(14, dtype=np.float32)),
            ), patch.object(
                eval_g1_client, "initialize_robot_to_starting_pose"
            ) as move_to_pose, patch.object(
                eval_g1_client.time, "sleep"
            ):
                eval_g1_client.eval_policy_client(fake_cfg(task="explicit task", max_steps=1), remote_policy)

            np.testing.assert_array_equal(move_to_pose.call_args_list[0].args[2], obs_pose)
            np.testing.assert_array_equal(move_to_pose.call_args_list[1].args[2], obs_pose)
            np.testing.assert_array_equal(move_to_pose.call_args_list[2].args[2], init_pose)
            self.assertEqual(
                [call_args.args[3] for call_args in move_to_pose.call_args_list],
                [False, False, False],
            )
            self.assertTrue(image_client.closed)

    def test_restores_from_obs_state_to_init_state_after_loop_exception(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            init_state_path = Path(temp_dir) / "init_state.json"
            obs_pose = np.arange(14, dtype=np.float32) + 10
            init_pose = np.arange(14, dtype=np.float32)
            obs_state_path.write_text(json.dumps({"observation.state": obs_pose.tolist()}), encoding="utf-8")
            init_state_path.write_text(json.dumps({"observation.state": init_pose.tolist()}), encoding="utf-8")
            remote_policy = FakeRemotePolicy()

            image_client = FakeImageClient()
            with patch.object(eval_g1_client, "OBS_STATE_PATH", obs_state_path), patch.object(
                eval_g1_client, "INIT_STATE_PATH", init_state_path, create=True
            ), patch.object(eval_g1_client, "setup_image_client", return_value=(image_client, {})), patch.object(
                eval_g1_client, "setup_robot_interface", return_value=fake_robot_interface()
            ), patch(
                "builtins.input", return_value="s"
            ), patch.object(
                eval_g1_client,
                "process_images_and_observations",
                side_effect=RuntimeError("loop boom"),
            ), patch.object(
                eval_g1_client, "initialize_robot_to_starting_pose"
            ) as move_to_pose, patch.object(
                eval_g1_client.time, "sleep"
            ):
                eval_g1_client.eval_policy_client(fake_cfg(task="explicit task", max_steps=1), remote_policy)

            np.testing.assert_array_equal(move_to_pose.call_args_list[0].args[2], obs_pose)
            np.testing.assert_array_equal(move_to_pose.call_args_list[1].args[2], obs_pose)
            np.testing.assert_array_equal(move_to_pose.call_args_list[2].args[2], init_pose)
            self.assertTrue(image_client.closed)


if __name__ == "__main__":
    unittest.main()
