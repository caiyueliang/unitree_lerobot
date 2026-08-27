import os
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_DDS_INTERFACE = "eth0"
LEGACY_DDS_INTERFACE = "enx6c1ff7bccc28"
DDS_INTERFACE_ENV = "UNITREE_DDS_INTERFACE"


def list_network_interfaces(sys_class_net: str | Path = "/sys/class/net") -> list[str]:
    return sorted(path.name for path in Path(sys_class_net).iterdir())


def resolve_dds_interface(interfaces: Iterable[str] | None = None) -> str:
    env_interface = os.environ.get(DDS_INTERFACE_ENV)
    if env_interface:
        return env_interface

    available = set(interfaces if interfaces is not None else list_network_interfaces())
    if DEFAULT_DDS_INTERFACE in available:
        return DEFAULT_DDS_INTERFACE
    if LEGACY_DDS_INTERFACE in available:
        return LEGACY_DDS_INTERFACE

    raise RuntimeError(
        f"No DDS network interface found. Set {DDS_INTERFACE_ENV} to the robot network interface name. "
        f"Available interfaces: {sorted(available)}"
    )


def initialize_dds_channel_factory(
    simulation_mode: bool,
    channel_factory_initialize: Callable[..., None],
) -> None:
    if simulation_mode:
        channel_factory_initialize(1)
    else:
        channel_factory_initialize(0, resolve_dds_interface())
