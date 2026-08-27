# obs_state.json 初始姿态缓存设计

## 背景

`unitree_lerobot/eval_robot/eval_g1_client.py` 的 `eval_policy_client` 当前每次启动真机评估时，都会读取数据集第 0 个 episode 的第一帧，并从 `observation.state` 或左右臂拆分状态中生成双臂初始姿态 `init_arm_pose`。用户希望把这个值缓存到本地，后续运行优先复用本地缓存。

## 目标

在 `eval_policy_client` 内直接实现 `obs_state.json` 的读写逻辑：

- 固定文件路径：`/home/unitree/caiyueliang/unitree_lerobot/obs_state.json`
- 文件存在时：读取 JSON 中保存的双臂初始姿态作为 `init_arm_pose`
- 文件不存在时：保持现有行为，从数据集第一帧获取初始姿态，并保存到 `obs_state.json`
- 如果 `cfg.task` 为空，仍从数据集第一帧读取任务文本作为 fallback

## 方案

采用方案 A：在 `eval_policy_client` 中内联读写 JSON。

实现方式：

1. 在模块顶部引入 `json` 和 `Path`。
2. 在模块级定义 `OBS_STATE_PATH = Path("/home/unitree/caiyueliang/unitree_lerobot/obs_state.json")`。
3. 在当前读取数据集第一帧的位置调整逻辑：
   - 如果 `obs_state.json` 存在，读取 JSON，转成 `np.float32` 一维数组，并校验长度等于 `arm_dof`。
   - 如果 `obs_state.json` 不存在，从数据集 step 调用 `_get_initial_arm_pose(step, arm_dof)`，并将 `init_arm_pose.tolist()` 写入 JSON。
4. JSON 格式使用简单结构：

```json
{
  "observation.state": [0.0, 0.0]
}
```

实际数组长度应等于运行时 `arm_dof`，例如 G1 双臂通常是 14。

## 错误处理

如果 `obs_state.json` 存在但内容无法解析、缺少 `observation.state` 字段，或数组长度不等于 `arm_dof`，直接抛出明确错误，不静默回退到数据集。原因是这个值会用于真机初始姿态，静默回退可能导致不可预期的机器人动作。

## 测试

新增或扩展针对 `eval_g1_client.py` 的测试，覆盖：

- 文件不存在时，从数据集 step 获取初始姿态，并写入 JSON。
- 文件存在时，优先使用 JSON 中的初始姿态。
- JSON 维度不等于 `arm_dof` 时抛出错误。

不运行真实机器人命令；测试只覆盖数据读取、缓存写入和维度校验。
