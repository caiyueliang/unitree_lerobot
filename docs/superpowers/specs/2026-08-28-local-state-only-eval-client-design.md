# Local State Only Eval Client Design

## Goal

`unitree_lerobot/eval_robot/eval_g1_client.py` should run the remote VLA real-robot client without loading a LeRobot dataset. The initial arm pose must come from `./obs_state.json`, and the task text must come from `--task`.

## Behavior

- `./obs_state.json` is required for the eval client startup pose. It must contain `{"observation.state": [...]}` with exactly the active arm DOF count.
- `--task` is required. The client will no longer read the task from dataset metadata.
- `--repo_id` is no longer required for this remote-policy client path.
- The client should fail before `remote_policy.reset()` if either `./obs_state.json` is missing/invalid or `--task` is blank.
- `./init_state.json` remains separate. It is used by the arm controller constructor as the initial DDS publish target before user confirmation.

## Architecture

Remove the dataset dependency from `eval_g1_client.py` only. Keep shared config fields compatible with other scripts by making `repo_id` default to an empty string in `EvalRealConfig`.

`eval_main` connects to the remote policy server and calls `eval_policy_client(cfg, remote_policy)`. `eval_policy_client` initializes image and robot interfaces, validates the explicit task, loads `./obs_state.json`, resets the remote policy, waits for `s`, moves the robot to the local initial pose, then enters the live observation/action loop.

## Testing

Use focused unit tests for the no-dataset contract:

- `eval_main` must not construct `LeRobotDataset`.
- `eval_policy_client` must use `./obs_state.json` and not accept a dataset argument.
- missing `./obs_state.json` stops before `remote_policy.reset()`.
- blank `--task` stops before `remote_policy.reset()`.
