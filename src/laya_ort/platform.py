"""声明这一版实际支持的运行目标。"""

import platform
import sys


def is_linux_x86_64() -> bool:
    """Linux 上的 x86_64，包含 Intel 和 AMD。"""
    machine = platform.machine().lower()
    return sys.platform == "linux" and machine in {"x86_64", "amd64"}


def describe_host() -> str:
    """当前解释器所在的系统和机器。"""
    return "%s %s" % (sys.platform, platform.machine())
