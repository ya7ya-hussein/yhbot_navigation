# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script to verify YH-Bot robot configuration loads correctly."""

from isaaclab.app import AppLauncher

# Create app launcher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Now import after app is created
from yhbot_navigation.tasks.manager_based.yhbot_navigation.assets.yh_bot_cfg import YHBOT_CFG

print("\n" + "="*80)
print("SUCCESS! Robot configuration loaded successfully!")
print("="*80)
print("\nRobot Configuration:")
print(f"  USD Path: {YHBOT_CFG.spawn.usd_path}")
print(f"  Controllable Joints: {list(YHBOT_CFG.actuators.keys())}")
print(f"  Initial Position: {YHBOT_CFG.init_state.pos}")
print(f"  Contact Sensors: {YHBOT_CFG.spawn.activate_contact_sensors}")
print("="*80 + "\n")

# Close the app
simulation_app.close()
