"""
Isaac Sim Script Editor Version
================================
Run this script directly in Isaac Sim's Script Editor (Window > Script Editor)

This script will enumerate all prims in the currently loaded warehouse USD file.
Make sure to load the warehouse USD first via File > Open or File > Add Reference.

Usage:
1. Load warehouse USD file in Isaac Sim
2. Open Script Editor (Window > Script Editor)
3. Copy and paste this script
4. Click "Run"
"""

import omni.usd
from pxr import Usd, UsdGeom
from typing import List, Dict


def analyze_current_stage():
    """Analyze the currently active USD stage in Isaac Sim."""
    
    # Get the current USD context
    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    
    if not stage:
        print("ERROR: No stage is currently loaded!")
        print("Please load a USD file first via File > Open")
        return
    
    print("=" * 80)
    print("Isaac Sim USD Stage Analysis")
    print("=" * 80)
    print(f"Root layer: {stage.GetRootLayer().identifier}")
    print()
    
    # Collect prim information
    all_prims = []
    mesh_prims = []
    xform_prims = []
    
    print("Traversing stage...")
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_type = prim.GetTypeName()
        
        all_prims.append({
            'path': prim_path,
            'type': prim_type,
            'depth': prim_path.count('/')
        })
        
        # Collect mesh prims specifically
        if prim.IsA(UsdGeom.Mesh):
            mesh_prims.append(prim_path)
        
        # Collect Xform prims
        if prim.IsA(UsdGeom.Xform):
            xform_prims.append(prim_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total prims: {len(all_prims)}")
    print(f"Mesh prims: {len(mesh_prims)}")
    print(f"Xform prims: {len(xform_prims)}")
    print()
    
    # Print first 50 prims in hierarchy
    print("\n" + "=" * 80)
    print("PRIM HIERARCHY (first 50):")
    print("=" * 80)
    for prim_info in all_prims[:50]:
        indent = "  " * (prim_info['depth'] - 1)
        name = prim_info['path'].split('/')[-1] or '/'
        type_info = f" [{prim_info['type']}]" if prim_info['type'] else ""
        print(f"{indent}{name}{type_info}")
    
    if len(all_prims) > 50:
        print(f"... and {len(all_prims) - 50} more prims")
    
    # Print mesh prims specifically (useful for MultiMeshRayCasterCfg)
    print("\n" + "=" * 80)
    print("MESH PRIMS (for MultiMeshRayCasterCfg):")
    print("=" * 80)
    print(f"Found {len(mesh_prims)} mesh prims\n")
    
    if mesh_prims:
        print("mesh_prim_paths = [")
        print('    "/World/ground",  # Add your ground plane')
        
        # Show first 20 mesh prims
        for mesh_path in mesh_prims[:20]:
            print(f'    "{mesh_path}",')
        
        if len(mesh_prims) > 20:
            print(f'    # ... and {len(mesh_prims) - 20} more mesh prims')
        
        print("]")
        
        # Also print with ENV_REGEX_NS for multi-environment setups
        print("\n# For multi-environment setup (with ENV_REGEX_NS):")
        print("mesh_prim_paths = [")
        print('    "/World/ground",  # Ground is usually shared')
        for mesh_path in mesh_prims[:10]:
            print(f'    "{{ENV_REGEX_NS}}{mesh_path}",')
        if len(mesh_prims) > 10:
            print(f'    # ... and {len(mesh_prims) - 10} more mesh prims')
        print("]")
    else:
        print("No mesh prims found!")
    
    # Print specific prim types you might be interested in
    print("\n" + "=" * 80)
    print("PRIM TYPE BREAKDOWN:")
    print("=" * 80)
    
    type_counts = {}
    for prim_info in all_prims:
        prim_type = prim_info['type'] or 'NoType'
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
    
    for prim_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"{prim_type:30s}: {count:4d}")
    
    print("\n" + "=" * 80)
    print("TIP: You can copy the mesh_prim_paths list above and use it in your")
    print("     yhbot_warehouse_env_cfg.py for the MultiMeshRayCasterCfg configuration.")
    print("=" * 80)
    
    return all_prims, mesh_prims


# Run the analysis
if __name__ == "__main__":
    all_prims, mesh_prims = analyze_current_stage()
