import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_initialize_robot_to_starting_pose_sends_pose_with_tau(self):
        arm_ctrl = Mock()
        arm_ik = Mock()
        pose = np.arange(14, dtype=np.float32)
        tau = np.arange(14, dtype=np.float32) + 100
        arm_ik.solve_tau.return_value = tau

        with patch.object(robot_arm.time, "sleep") as sleep:
            robot_arm.initialize_robot_to_starting_pose(arm_ctrl, arm_ik, pose, send_real_robot=True, wait_s=0.5)

        arm_ik.solve_tau.assert_called_once_with(pose)
        arm_ctrl.ctrl_dual_arm.assert_called_once_with(pose, tau)
        sleep.assert_called_once_with(0.5)


if __name__ == "__main__":
    unittest.main()
