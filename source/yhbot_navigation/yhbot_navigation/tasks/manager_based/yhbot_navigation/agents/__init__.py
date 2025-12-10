# /home/yahya/Documents/fyp/yhbot_navigation/source/yhbot_navigation/yhbot_navigation/tasks/manager_based/yhbot_navigation/agents/__init__.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

# Convenience variable for the directory where agent configs are stored
SKRL_AGENTS_CFG_DIR = os.path.join(os.path.dirname(__file__))

__all__ = ["SKRL_AGENTS_CFG_DIR"]