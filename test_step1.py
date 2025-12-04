#!/usr/bin/env python3
"""
Step 1 Testing Script - Configuration Validation
Tests scene configuration without requiring environment registration
"""

import sys
import os

print("=" * 60)
print("STEP 1 - Scene Configuration Test")
print("=" * 60)

# Test 1: Import Configuration
print("\n[Test 1/4] Testing configuration import...")
try:
    from yhbot_navigation.tasks.manager_based.yhbot_navigation.yhbot_navigation_env_cfg import (
        YhbotNavigationEnvCfg,
        YhbotNavigationSceneCfg
    )
    print("✅ Configuration imports successfully")
except Exception as e:
    print(f"❌ Failed to import configuration: {e}")
    sys.exit(1)

# Test 2: Create Configuration Instance
print("\n[Test 2/4] Testing configuration instantiation...")
try:
    env_cfg = YhbotNavigationEnvCfg()
    print("✅ Configuration instantiated successfully")
except Exception as e:
    print(f"❌ Failed to instantiate configuration: {e}")
    sys.exit(1)

# Test 3: Verify Parameters
print("\n[Test 3/4] Verifying configuration parameters...")
try:
    assert env_cfg.scene.num_envs == 4096, f"Expected 4096 envs, got {env_cfg.scene.num_envs}"
    assert env_cfg.scene.env_spacing == 4.0, f"Expected 4.0m spacing, got {env_cfg.scene.env_spacing}"
    assert env_cfg.decimation == 4, f"Expected decimation 4, got {env_cfg.decimation}"
    assert env_cfg.episode_length_s == 120.0, f"Expected 120s episodes, got {env_cfg.episode_length_s}"
    assert env_cfg.sim.dt == 1/60, f"Expected dt=1/60, got {env_cfg.sim.dt}"
    print("✅ All parameters correct")
    print(f"   - num_envs: {env_cfg.scene.num_envs}")
    print(f"   - env_spacing: {env_cfg.scene.env_spacing}m")
    print(f"   - decimation: {env_cfg.decimation} (15 Hz policy)")
    print(f"   - episode_length: {env_cfg.episode_length_s}s")
    print(f"   - physics_dt: {env_cfg.sim.dt}s (60 Hz)")
except AssertionError as e:
    print(f"❌ Parameter verification failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

# Test 4: Check Robot Configuration
print("\n[Test 4/4] Checking robot configuration...")
try:
    from yhbot_navigation.tasks.manager_based.yhbot_navigation.assets.yh_bot_cfg import YHBOT_CFG
    
    # Check robot is properly configured
    assert YHBOT_CFG is not None, "YHBOT_CFG is None"
    assert hasattr(YHBOT_CFG, 'spawn'), "YHBOT_CFG missing spawn config"
    
    # Check USD path
    usd_path = YHBOT_CFG.spawn.usd_path
    print(f"   - USD path: {usd_path}")
    
    # Check if USD file exists (if path is absolute)
    if os.path.isabs(usd_path) and os.path.exists(usd_path):
        print(f"   - USD file exists: ✅")
    elif not os.path.isabs(usd_path):
        print(f"   - USD path is relative (good for portability): ✅")
    
    # Check contact sensors are enabled
    assert YHBOT_CFG.spawn.activate_contact_sensors == True, "Contact sensors not enabled"
    print("   - Contact sensors enabled: ✅")
    
    print("✅ Robot configuration valid")
except Exception as e:
    print(f"❌ Robot configuration check failed: {e}")
    sys.exit(1)

# Final Summary
print("\n" + "=" * 60)
print("✅ STEP 1 VALIDATION COMPLETE")
print("=" * 60)
print("\n📋 Summary:")
print("   ✅ Configuration imports successfully")
print("   ✅ Configuration instantiates correctly")
print("   ✅ All parameters have correct values")
print("   ✅ Robot configuration is valid")
print("\n🎯 Step 1 Status: PASSED")
print("\n⚠️  Note: Full environment testing requires:")
print("   - Step 2: Sensor configuration")
print("   - Step 3: Action configuration")
print("   - Steps 4-12: Complete MDP setup")
print("   - Step 13: Environment registration")
print("\n📝 Once Step 13 is complete, you can run:")
print("   python scripts/zero_agent.py --task Isaac-Navigation-Grid-YHBot-v0 --num_envs 10")
print("\n✨ Ready to proceed to Step 2!")
print("=" * 60)
