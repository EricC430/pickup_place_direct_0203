#!/usr/bin/env python3
"""[0410 ANTI-PENETRATION] Strategy 3: Enhance Convex Decomposition parameters on USD objects.

This script opens each object USD file and supplements the existing convex decomposition
collision meshes with tighter parameters to prevent thin-object penetration:

- min_thickness: Ensures colliders have a minimum thickness (prevents ultra-thin geometry from
  being ignored by the collision detection)
- shrink_wrap: Projects convex hull vertices onto the original mesh surface for tighter fit
- error_percentage: Controls how closely the decomposition approximates the original geometry
- voxel_resolution: Controls the voxel grid resolution used for decomposition

Usage:
    # Dry-run (preview changes, no file modification):
    python scripts/enhance_convex_decomposition.py --dry-run

    # Apply changes:
    python scripts/enhance_convex_decomposition.py

    # Custom USD base path:
    python scripts/enhance_convex_decomposition.py --base-path /root/ObjectFolder

    # Specific object IDs:
    python scripts/enhance_convex_decomposition.py --object-ids 62 39 41
"""

import argparse
import sys
import os

# Add the pxr module path for USD operations
try:
    from pxr import Usd, UsdPhysics, PhysxSchema, UsdGeom
except ImportError:
    print("ERROR: pxr module not found. This script must be run inside the Isaac Sim container where USD Python is available.")
    print("  e.g.: python scripts/enhance_convex_decomposition.py")
    sys.exit(1)


# Default object IDs matching the env_cfg
DEFAULT_OBJECT_IDS = [62]  # [39, 41, 68, 25]

# Default convex decomposition parameters optimized for anti-penetration
DEFAULT_PARAMS = {
    "min_thickness": 0.005,       # 5mm minimum thickness (default is 0.001)
    "shrink_wrap": True,           # Enable surface projection for tighter collider fit
    "error_percentage": 5.0,       # 5% error tolerance (default is 10%)
    "hull_vertex_limit": 64,       # Max vertices per convex hull
    "max_convex_hulls": 32,        # Max number of convex hulls
    "voxel_resolution": 500000,    # Voxel resolution for decomposition
}


def find_mesh_prims_with_collision(stage):
    """Find all mesh prims that have collision API applied."""
    mesh_prims = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh_prims.append(prim)
    return mesh_prims


def get_collision_approximation(prim):
    """Get the collision approximation type from a prim."""
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
        approx_attr = mesh_collision_api.GetApproximationAttr()
        if approx_attr and approx_attr.IsValid():
            return approx_attr.Get()
    return None


def enhance_convex_decomposition(prim, params, dry_run=False):
    """Enhance convex decomposition parameters on a prim.
    
    Returns True if changes were made/would be made.
    """
    prim_path = str(prim.GetPath())
    
    # Check if this prim has convex decomposition
    approx = get_collision_approximation(prim)
    
    if approx != "convexDecomposition":
        # Check if PhysxConvexDecompositionCollisionAPI is applied directly
        if not prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
            return False
    
    print(f"  Found convex decomposition on: {prim_path}")
    
    # Get or apply the PhysxConvexDecompositionCollisionAPI
    if prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
        decomp_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
    else:
        if dry_run:
            print(f"    [DRY-RUN] Would apply PhysxConvexDecompositionCollisionAPI")
            decomp_api = None
        else:
            decomp_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    
    changes = []
    
    if decomp_api is not None:
        # Read current values and compare
        attr_map = {
            "min_thickness": ("GetMinThicknessAttr", "CreateMinThicknessAttr"),
            "shrink_wrap": ("GetShrinkWrapAttr", "CreateShrinkWrapAttr"),
            "error_percentage": ("GetErrorPercentageAttr", "CreateErrorPercentageAttr"),
            "hull_vertex_limit": ("GetHullVertexLimitAttr", "CreateHullVertexLimitAttr"),
            "max_convex_hulls": ("GetMaxConvexHullsAttr", "CreateMaxConvexHullsAttr"),
            "voxel_resolution": ("GetVoxelResolutionAttr", "CreateVoxelResolutionAttr"),
        }
        
        for param_name, (getter_name, setter_name) in attr_map.items():
            desired_value = params[param_name]
            getter_func = getattr(decomp_api, getter_name, None)
            setter_func = getattr(decomp_api, setter_name, None)
            
            current_value = None
            if getter_func:
                attr = getter_func()
                if attr and attr.IsValid():
                    current_value = attr.Get()
            
            if current_value != desired_value:
                changes.append(f"    {param_name}: {current_value} → {desired_value}")
                if not dry_run and setter_func:
                    setter_func(desired_value).Set(desired_value) if current_value is None else getter_func().Set(desired_value)
    else:
        # dry_run with no API yet
        for param_name, desired_value in params.items():
            changes.append(f"    {param_name}: (unset) → {desired_value}")
    
    if changes:
        prefix = "[DRY-RUN] " if dry_run else ""
        print(f"    {prefix}Changes:")
        for c in changes:
            print(f"      {c}")
    else:
        print(f"    No changes needed (already configured)")
    
    return len(changes) > 0


def process_usd_file(usd_path, params, dry_run=False):
    """Process a single USD file to enhance convex decomposition parameters."""
    print(f"\n{'='*60}")
    print(f"Processing: {usd_path}")
    
    if not os.path.exists(usd_path):
        print(f"  WARNING: File not found: {usd_path}")
        return False
    
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as e:
        print(f"  ERROR: Failed to open USD: {e}")
        return False
    
    mesh_prims = find_mesh_prims_with_collision(stage)
    
    if not mesh_prims:
        print(f"  WARNING: No mesh prims with collision found")
        return False
    
    print(f"  Found {len(mesh_prims)} mesh/collision prims")
    
    any_changes = False
    for prim in mesh_prims:
        if enhance_convex_decomposition(prim, params, dry_run):
            any_changes = True
    
    if any_changes and not dry_run:
        try:
            stage.Save()
            print(f"  ✅ Saved changes to {usd_path}")
        except Exception as e:
            print(f"  ERROR: Failed to save: {e}")
            return False
    elif not any_changes:
        print(f"  ℹ️  No changes needed")
    
    return any_changes


def main():
    parser = argparse.ArgumentParser(
        description="Enhance convex decomposition collision parameters on object USD files"
    )
    parser.add_argument(
        "--base-path",
        default="/workspace/test_isaaclab/ObjectFolder_selected",
        help="Base path to ObjectFolder directory"
    )
    parser.add_argument(
        "--object-ids",
        nargs="+",
        type=int,
        default=None,
        help="Object IDs to process (default: use DEFAULT_OBJECT_IDS)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--min-thickness",
        type=float,
        default=DEFAULT_PARAMS["min_thickness"],
        help=f"Minimum thickness for convex hulls (default: {DEFAULT_PARAMS['min_thickness']})"
    )
    parser.add_argument(
        "--error-percentage",
        type=float,
        default=DEFAULT_PARAMS["error_percentage"],
        help=f"Error percentage for decomposition (default: {DEFAULT_PARAMS['error_percentage']})"
    )
    parser.add_argument(
        "--no-shrink-wrap",
        action="store_true",
        help="Disable shrink wrap (default: enabled)"
    )
    args = parser.parse_args()
    
    object_ids = args.object_ids or DEFAULT_OBJECT_IDS
    params = dict(DEFAULT_PARAMS)
    params["min_thickness"] = args.min_thickness
    params["error_percentage"] = args.error_percentage
    params["shrink_wrap"] = not args.no_shrink_wrap
    
    print("=" * 60)
    print("Convex Decomposition Enhancement Script")
    print("=" * 60)
    print(f"Mode: {'DRY-RUN (no changes)' if args.dry_run else 'APPLY CHANGES'}")
    print(f"Base path: {args.base_path}")
    print(f"Object IDs: {object_ids}")
    print(f"Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    total_changed = 0
    total_processed = 0
    
    for obj_id in object_ids:
        usd_path = os.path.join(args.base_path, str(obj_id), f"{obj_id}.usd")
        total_processed += 1
        if process_usd_file(usd_path, params, args.dry_run):
            total_changed += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {total_changed}/{total_processed} files {'would be ' if args.dry_run else ''}modified")
    if args.dry_run and total_changed > 0:
        print(f"\nRe-run without --dry-run to apply changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
