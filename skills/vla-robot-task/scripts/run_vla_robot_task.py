#!/usr/bin/env python3
"""Run supported Unitree VLA real-robot tasks."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys


TASKS = {
    "test_tubes": "Place the test tubes neatly back into the test tube rack.",
    "fruit_sorting": "Place each fruit onto the plate with the matching color.",
}

PROJECT_DIR = "/home/unitree/caiyueliang/unitree_lerobot"
PYTHONPATH_PARTS = [
    "/home/unitree/caiyueliang/lerobot/src",
    "/home/unitree/caiyueliang/unitree_lerobot",
    "/home/unitree/roboclaw/robot/teleimager/src",
]
SUBMISSION_TOKEN = "dENyPN2dPlW6AWcaospx6ueYpcxjCeAz"
DDS_INTERFACE = "eth0"


def build_command(task: str) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "lerobot_cyl",
        "python",
        "unitree_lerobot/eval_robot/eval_g1_client.py",
        f"--task={task}",
        "--policy_server_uri=ws://192.168.123.2:8765",
        "--episodes=0",
        "--frequency=30",
        "--max_steps=1800",
        "--arm=G1_29",
        "--ee=dex1",
        "--image_host=127.0.0.1",
        "--visualization=false",
        "--send_real_robot=true",
        "--motion=true",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-key", choices=sorted(TASKS), required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    task = TASKS[args.task_key]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = ":".join(PYTHONPATH_PARTS + ([existing_pythonpath] if existing_pythonpath else []))
    env["UNIBOT_SUBMISSION_TOKEN"] = env.get("UNIBOT_SUBMISSION_TOKEN", SUBMISSION_TOKEN)
    env["UNITREE_DDS_INTERFACE"] = env.get("UNITREE_DDS_INTERFACE", DDS_INTERFACE)

    command = build_command(task)
    if args.dry_run:
        print("cd", PROJECT_DIR)
        print("PYTHONPATH=" + env["PYTHONPATH"])
        print("UNIBOT_SUBMISSION_TOKEN=" + env["UNIBOT_SUBMISSION_TOKEN"])
        print("UNITREE_DDS_INTERFACE=" + env["UNITREE_DDS_INTERFACE"])
        print(shlex.join(command))
        return 0

    return subprocess.run(command, cwd=PROJECT_DIR, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
