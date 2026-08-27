import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from unitree_lerobot.eval_robot import eval_g1_client


class FakeDataset:
    def __init__(self, step):
        self.step = step
        self.read_count = 0
        self.meta = SimpleNamespace(episodes={"dataset_from_index": [0]})

    def __getitem__(self, index):
        self.read_count += 1
        assert index == 0
        return self.step


class FakeRemotePolicy:
    metadata = {}

    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        return {"ok": True}


class FakeImageClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def fake_cfg(task="cached task"):
    return SimpleNamespace(visualization=False, task=task, send_real_robot=False, frequency=30.0, ee="")


def fake_robot_interface():
    return {
        "arm_ctrl": object(),
        "arm_ik": object(),
        "ee_shared_mem": {},
        "arm_dof": 14,
        "ee_dof": 0,
    }


class ObsStateCacheTest(unittest.TestCase):
    def run_client_until_prompt(self, cfg, dataset, remote_policy, obs_state_path):
        image_client = FakeImageClient()
        with patch.object(eval_g1_client, "OBS_STATE_PATH", obs_state_path), patch.object(
            eval_g1_client, "setup_image_client", return_value=(image_client, {})
        ), patch.object(eval_g1_client, "setup_robot_interface", return_value=fake_robot_interface()), patch(
            "builtins.input", return_value=""
        ):
            eval_g1_client.eval_policy_client(cfg, dataset, remote_policy)
        self.assertTrue(image_client.closed)

    def test_missing_obs_state_cache_is_saved_from_dataset_first_frame(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            dataset_state = np.arange(20, dtype=np.float32)
            dataset = FakeDataset({"task": "dataset task", "observation.state": dataset_state})
            remote_policy = FakeRemotePolicy()

            self.run_client_until_prompt(fake_cfg(), dataset, remote_policy, obs_state_path)

            with obs_state_path.open("r", encoding="utf-8") as file:
                saved = json.load(file)
            self.assertEqual(saved["observation.state"], np.arange(14, dtype=np.float32).tolist())
            self.assertEqual(dataset.read_count, 1)
            self.assertEqual(remote_policy.reset_count, 1)

    def test_existing_obs_state_cache_is_used_without_reading_dataset_when_task_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            cached_state = [float(value) for value in range(100, 114)]
            obs_state_path.write_text(json.dumps({"observation.state": cached_state}), encoding="utf-8")
            dataset = FakeDataset({"task": "dataset task", "observation.state": np.arange(14, dtype=np.float32)})
            remote_policy = FakeRemotePolicy()

            self.run_client_until_prompt(fake_cfg(task="explicit task"), dataset, remote_policy, obs_state_path)

            self.assertEqual(dataset.read_count, 0)
            self.assertEqual(remote_policy.reset_count, 1)
            self.assertEqual(json.loads(obs_state_path.read_text(encoding="utf-8"))["observation.state"], cached_state)

    def test_invalid_obs_state_cache_stops_before_remote_policy_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            obs_state_path = Path(temp_dir) / "obs_state.json"
            obs_state_path.write_text(json.dumps({"observation.state": [1.0, 2.0]}), encoding="utf-8")
            dataset = FakeDataset({"task": "dataset task", "observation.state": np.arange(14, dtype=np.float32)})
            remote_policy = FakeRemotePolicy()

            self.run_client_until_prompt(fake_cfg(task="explicit task"), dataset, remote_policy, obs_state_path)

            self.assertEqual(dataset.read_count, 0)
            self.assertEqual(remote_policy.reset_count, 0)


if __name__ == "__main__":
    unittest.main()
