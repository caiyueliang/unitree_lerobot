# Local State Only Eval Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove dataset loading from the remote VLA eval client and require local `obs_state.json` plus explicit `--task`.

**Architecture:** Keep the change scoped to `eval_g1_client.py`, its focused tests, and the shared config default. `eval_main` becomes remote-policy-only, and `eval_policy_client` loads the local initial arm pose directly.

**Tech Stack:** Python, unittest, unittest.mock, numpy, LeRobot parser dataclass config.

---

### Task 1: Define No-Dataset Client Contract

**Files:**
- Modify: `test/test_eval_g1_client_obs_state_cache.py`

- [x] **Step 1: Write failing tests**

Add tests that prove `eval_main` does not construct `LeRobotDataset`, `eval_policy_client` loads local `obs_state.json`, and missing local state or blank task stops before policy reset.

- [x] **Step 2: Run tests to verify failure**

Run: `conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=.:/home/unitree/caiyueliang/unitree_lerobot/unitree_lerobot/eval_robot python test/test_eval_g1_client_obs_state_cache.py -v`

Expected before implementation: failures caused by the old dataset argument and unconditional dataset load.

### Task 2: Remove Dataset Dependency From Eval Client

**Files:**
- Modify: `unitree_lerobot/eval_robot/eval_g1_client.py`
- Modify: `unitree_lerobot/eval_robot/utils/utils.py`

- [x] **Step 1: Make `repo_id` optional for this command parser**

Change `EvalRealConfig.repo_id` from a required string to `repo_id: str = ""`.

- [x] **Step 2: Remove dataset import and first-frame fallback from `eval_g1_client.py`**

Delete `LeRobotDataset`, `list_unique_tasks`, `_to_numpy_1d`, and `_get_initial_arm_pose` from the remote client.

- [x] **Step 3: Add local state loader**

Add `_load_initial_arm_pose(arm_dof, path=OBS_STATE_PATH)` that validates the local JSON file and returns a flat `np.float32` vector.

- [x] **Step 4: Update `eval_policy_client` signature and startup**

Change it to `eval_policy_client(cfg, remote_policy)`, require non-blank `cfg.task`, load the local pose, and stop before `remote_policy.reset()` on validation errors.

- [x] **Step 5: Update `eval_main`**

Remove `LeRobotDataset` construction and metadata logging. Connect to `RemotePolicy` and call `eval_policy_client(cfg, remote_policy)`.

### Task 3: Verify

**Files:**
- Test: `test/test_eval_g1_client_obs_state_cache.py`
- Test: `test/test_robot_arm_init_state.py`

- [x] **Step 1: Run focused client tests**

Run: `conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=.:/home/unitree/caiyueliang/unitree_lerobot/unitree_lerobot/eval_robot python test/test_eval_g1_client_obs_state_cache.py -v`

Expected: all tests pass.

- [x] **Step 2: Run controller init-state tests**

Run: `conda run --no-capture-output -n lerobot_cyl env PYTHONPATH=. python test/test_robot_arm_init_state.py -v`

Expected: all tests pass.

- [x] **Step 3: Compile changed modules**

Run: `conda run --no-capture-output -n lerobot_cyl python -m py_compile unitree_lerobot/eval_robot/eval_g1_client.py unitree_lerobot/eval_robot/utils/utils.py test/test_eval_g1_client_obs_state_cache.py`

Expected: exit code 0.
