# Preserve Arm State Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop reordering split `observation.state.left_arm` and `observation.state.right_arm` values when building the initial dual-arm pose.

**Architecture:** Keep the change local to `unitree_lerobot/eval_robot/eval_g1_client.py`. Flat `observation.state` datasets keep the existing behavior, while split left/right arm datasets are converted to 1-D arrays and concatenated in their original dataset order.

**Tech Stack:** Python, NumPy, `unittest`, `conda run`.

---

### Task 1: Preserve Split Arm State Order

**Files:**
- Modify: `unitree_lerobot/eval_robot/eval_g1_client.py`
- Modify: `test/test_eval_g1_client_obs_state_cache.py`

- [ ] **Step 1: Write the failing test**

Add this test to `test/test_eval_g1_client_obs_state_cache.py`:

```python
    def test_initial_arm_pose_preserves_split_arm_dataset_order(self):
        step = {
            "observation.state.left_arm": np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.float32),
            "observation.state.right_arm": np.array([20, 21, 22, 23, 24, 25, 26], dtype=np.float32),
        }

        init_arm_pose = eval_g1_client._get_initial_arm_pose(step, arm_dof=14)

        np.testing.assert_array_equal(
            init_arm_pose,
            np.array([10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26], dtype=np.float32),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=.:/home/unitree/caiyueliang/unitree_lerobot/unitree_lerobot/eval_robot python test/test_eval_g1_client_obs_state_cache.py -v
```

Expected: the new test fails because the current code reorders each split arm with `STANDARD_ARM_JOINT_INDICES`.

- [ ] **Step 3: Write minimal implementation**

In `unitree_lerobot/eval_robot/eval_g1_client.py`, remove the `STANDARD_ARM_JOINT_INDICES` constant and `_standardize_split_arm` helper from the active code path. Replace the split state concatenation with:

```python
    init_arm_pose = np.concatenate(
        (
            _to_numpy_1d(step["observation.state.left_arm"]),
            _to_numpy_1d(step["observation.state.right_arm"]),
        )
    )
```

Keep the existing final `arm_dof` length check.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=.:/home/unitree/caiyueliang/unitree_lerobot/unitree_lerobot/eval_robot python test/test_eval_g1_client_obs_state_cache.py -v
```

Expected: all obs state cache tests pass.

- [ ] **Step 5: Run syntax verification**

Run:

```bash
conda run --no-capture-output -n lerobot_cyl python -m py_compile unitree_lerobot/eval_robot/eval_g1_client.py test/test_eval_g1_client_obs_state_cache.py
```

Expected: exit code 0.

---

### Self-Review

- Spec coverage: removes the split arm reordering and preserves original dataset order.
- Placeholder scan: no placeholders remain.
- Type consistency: split arm values remain NumPy float32 arrays produced by `_to_numpy_1d`.
