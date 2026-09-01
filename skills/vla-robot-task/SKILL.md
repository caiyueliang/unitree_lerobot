---
name: vla-robot-task
description: 调用 Unitree VLA 模型推理服务控制真机执行受支持的操作任务。当用户要求机器人或 VLA 策略执行抓取并放置试管、整理试管、水果分类、按颜色把水果放到对应盘子等场景时使用。
---

# VLA 机器人任务

仅在用户请求受支持的真机 VLA 推理任务时使用此技能。

## 支持的任务

将用户请求映射到且仅映射到一个受支持的英文任务文本：

| 用户意图 | `--task-key` 参数值 |
| --- | --- |
| 抓取并放置试管, 整理试管, 把试管放回试管架 | `test_tubes` |
| 水果分类, 按颜色分水果, 把水果放到同色盘子 | `fruit_sorting` |

如果用户请求的任务不匹配上面的受支持意图，必须精确回复：

```text
抱歉，我当前还不支持这个任务
```

对于不支持的任务，不要运行任何机器人命令。

## 运行流程

1. 判断用户请求是否匹配一个受支持任务。
2. 如果任务受支持，在此技能目录下运行 `scripts/run_vla_robot_task.py --task-key <key>`。
3. 在有帮助时向用户展示命令输出。
4. 如果命令失败，报告失败并包含关键错误行。

试管任务使用 `test_tubes`。水果分类任务使用 `fruit_sorting`。

脚本会使用以下固定参数运行真机命令：

- conda 环境：`lerobot_cyl`
- 项目目录：`/home/unitree/caiyueliang/unitree_lerobot`
- `PYTHONPATH` 前缀：`/home/unitree/caiyueliang/lerobot/src:/home/unitree/caiyueliang/unitree_lerobot:/home/unitree/roboclaw/robot/teleimager/src`
- 提交 token：优先使用环境变量 `UNIBOT_SUBMISSION_TOKEN`，未设置时使用脚本内置默认值
- DDS 网卡：优先使用环境变量 `UNITREE_DDS_INTERFACE`，未设置时使用 `eth0`
- 策略服务：`ws://192.168.123.2:8765`
- 图像主机：`127.0.0.1`
- 频率：`30`
- 最大步数：`1800`
- 机械臂：`G1_29`
- 末端执行器：`dex1`
- 启用真机发送
- 启用动作执行：`--motion=true`
- 禁用可视化：`--visualization=false`

除非用户明确要求，否则不要修改这些参数。
