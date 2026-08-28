# Controller Init State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `G1_29_ArmController` initialize its publish target from `./init_state.json` instead of `np.zeros(14)`.

**Architecture:** Keep the change local to `unitree_lerobot/eval_robot/robot_control/robot_arm.py`. Add a small JSON loader for the same `{"observation.state": [...]}` format used by the eval client, then call it before the controller starts its publish thread.

**Tech Stack:** Python, JSON, NumPy, `unittest`, `conda run`.

---

### Task 1: Load G1 Init Target From JSON

**Files:**
- Modify: `unitree_lerobot/eval_robot/robot_control/robot_arm.py`
- Create: `test/test_robot_arm_init_state.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_robot_arm_init_state.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unitree_lerobot.eval_robot.robot_control import robot_arm


class InitStateTest(unittest.TestCase):
    def test_load_initial_arm_q_target_reads_observation_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "init_state.json"
            values = [float(value) for value in range(14)]
            path.write_text(json.dumps({"observation.state": values}), encoding="utf-8")

            q_target = robot_arm._load_initial_arm_q_target(14, path)

            np.testing.assert_array_equal(q_target, np.array(values, dtype=np.float32))

    def test_load_initial_arm_q_target_rejects_wrong_dimension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "init_state.json"
            path.write_text(json.dumps({"observation.state": [0.0, 1.0]}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain 14 values"):
                robot_arm._load_initial_arm_q_target(14, path)

    def test_load_initial_arm_q_target_requires_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "init_state.json"

            with self.assertRaisesRegex(FileNotFoundError, "init_state.json"):
                robot_arm._load_initial_arm_q_target(14, path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=. python test/test_robot_arm_init_state.py -v
```

Expected: fail because `_load_initial_arm_q_target` does not exist yet.

- [ ] **Step 3: Implement loader and use it in G1_29 controller**

In `unitree_lerobot/eval_robot/robot_control/robot_arm.py`, add imports:

```python
import json
from pathlib import Path
```

Add module constant and helper:

```python
INIT_STATE_PATH = Path("./init_state.json")


def _load_initial_arm_q_target(num_joints, path=INIT_STATE_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create it with an 'observation.state' list.")
    with path.open("r", encoding="utf-8") as file:
        init_state_data = json.load(file)
    if "observation.state" not in init_state_data:
        raise ValueError(f"{path} must contain 'observation.state'.")
    q_target = np.asarray(init_state_data["observation.state"], dtype=np.float32).reshape(-1)
    if q_target.shape[0] != num_joints:
        raise ValueError(f"{path} observation.state must contain {num_joints} values, got {q_target.shape[0]}.")
    return q_target
```

Replace:

```python
self.q_target = np.zeros(14)
```

inside `G1_29_ArmController.__init__` with:

```python
self.q_target = _load_initial_arm_q_target(14)
```

Leave `ctrl_dual_arm_go_home()` unchanged because that explicit method is still a deliberate command to go home.

- [ ] **Step 4: Run tests**

Run:

```bash
conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=. python test/test_robot_arm_init_state.py -v
```

Expected: all tests pass.

Run:

```bash
conda run --no-capture-output -n lerobot_cyl python -m py_compile unitree_lerobot/eval_robot/robot_control/robot_arm.py test/test_robot_arm_init_state.py
```

Expected: exit code 0.

---

### Self-Review

- Spec coverage: `G1_29_ArmController` no longer starts with a zero arm target and instead reads `./init_state.json`.
- Placeholder scan: no placeholders remain.
- Type consistency: the loader returns a 1-D `np.float32` vector with exactly the requested joint count.
