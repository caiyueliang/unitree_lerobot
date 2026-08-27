import os
import unittest
from unittest.mock import patch


from unitree_lerobot.eval_robot.robot_control.dds_interface import resolve_dds_interface


class DdsInterfaceTest(unittest.TestCase):
    def test_resolve_dds_interface_uses_env_override(self):
        with patch.dict(os.environ, {"UNITREE_DDS_INTERFACE": "usb0"}):
            self.assertEqual(resolve_dds_interface(["eth0"]), "usb0")

    def test_resolve_dds_interface_defaults_to_eth0_when_available(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_dds_interface(["lo", "wlan0", "eth0"]), "eth0")


if __name__ == "__main__":
    unittest.main()
