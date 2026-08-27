# Obs State Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache the first-frame arm pose used by `eval_policy_client` in `./obs_state.json` relative to the command's working directory and prefer that local cache on later runs.

**Architecture:** Keep the implementation inline in `unitree_lerobot/eval_robot/eval_g1_client.py` as requested. The function will only read the dataset first frame when it needs the task fallback or when the cache file does not exist; invalid cache contents will stop before robot initialization.

**Tech Stack:** Python, NumPy, JSON, `unittest`, `unittest.mock`.

---

### Task 1: Add Obs State Cache Behavior

**Files:**
- Modify: `unitree_lerobot/eval_robot/eval_g1_client.py`
- Create: `test/test_eval_g1_client_obs_state_cache.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_eval_g1_client_obs_state_cache.py` with fake dataset, robot interface, image client, and remote policy objects so `eval_policy_client` can be exercised up to the start prompt without touching hardware.

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=.:/home/unitree/caiyueliang/unitree_lerobot/unitree_lerobot/eval_robot python test/test_eval_g1_client_obs_state_cache.py -v`

Expected: FAIL or ERROR because `eval_g1_client.OBS_STATE_PATH` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

In `unitree_lerobot/eval_robot/eval_g1_client.py`, add:

```python
import json
from pathlib import Path
```

Add near constants:

```python
OBS_STATE_PATH = Path("./obs_state.json")
```

Replace the dataset-first-frame initialization block in `eval_policy_client` with logic that:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=.:/home/unitree/caiyueliang/unitree_lerobot/unitree_lerobot/eval_robot python test/test_eval_g1_client_obs_state_cache.py -v`

Expected: all three tests pass.

- [ ] **Step 5: Run nearby regression tests**

Run: `conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=.:/home/unitree/caiyueliang/unitree_lerobot/unitree_lerobot/eval_robot python test/test_eval_g1_client_obs_state_cache.py -v`

Expected: PASS.

Run: `conda run --no-capture-output -n lerobot_cyl python -m py_compile unitree_lerobot/eval_robot/eval_g1_client.py`

Expected: exit code 0.

- [ ] **Step 6: Commit**

```bash
git add unitree_lerobot/eval_robot/eval_g1_client.py test/test_eval_g1_client_obs_state_cache.py docs/superpowers/plans/2026-08-27-obs-state-cache.md
git commit -m "feat: cache eval initial obs state"
```

---

### Self-Review

- Spec coverage: the cache path is fixed to `./obs_state.json` relative to the command's working directory; existing cache is preferred; missing cache is populated from dataset first frame; unsupported task behavior is not changed.
- Placeholder scan: no placeholders remain.
- Type consistency: `observation.state` is represented as a JSON list and converted to a 1-D `np.float32` array before robot initialization.
