# SPDX-License-Identifier: MIT
"""Build the Claim Observatory V3 modular Blender world.

Run from the repository root:

    /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python visual_identity/claim_observatory/create_claim_observatory_world_v3.py \
      -- --render-stills
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
EXPECTED_ROOT = REPO_ROOT / "examples/claim_governance_capsule/expected"
CAPSULE_ROOT = REPO_ROOT / "examples/claim_governance_capsule"
RENDER_DIR = ROOT / "renders"
BLEND_PATH = ROOT / "claim_observatory_world_v3.blend"
VISUAL_MANIFEST_PATH = ROOT / "visual_world_manifest_v3.json"

FPS = 24
FRAME_END = 660
CAPSULE_CENTER = Vector((0.0, 0.0, 1.55))

SOURCE_ARTIFACTS = {
    "claim_envelope": "examples/claim_governance_capsule/expected/claim_envelope.json",
    "claim_governance_audit": "examples/claim_governance_capsule/expected/claim_governance_audit.json",
    "cc_report": "examples/claim_governance_capsule/expected/cc_report.json",
    "manifest": "examples/claim_governance_capsule/manifest.expected.json",
    "bounds": "examples/claim_governance_capsule/expected/bounds.json",
    "decay_policy": "examples/claim_governance_capsule/expected/decay_policy.json",
    "extremal_lower": "examples/claim_governance_capsule/expected/extremal_lower.json",
    "extremal_upper": "examples/claim_governance_capsule/expected/extremal_upper.json",
    "confirmatory_protocol": "examples/claim_governance_capsule/expected/confirmatory_protocol.json",
    "confirmatory_failure_matrix": "examples/claim_governance_capsule/expected/confirmatory_failure_matrix.json",
}

COLLECTION_NAMES = [
    "00_WorldRoot",
    "01_ArrivalHall",
    "02_ClaimCapsuleChamber",
    "03_EvidenceVault",
    "04_SupportGraphOrrery",
    "05_NonClaimsWall",
    "06_DecayClockRoom",
    "07_ChallengeRange",
    "08_ReplayManifestEngine",
    "09_HumanReviewTribunal",
    "10_LedgerTower",
    "11_FrechetAtomGarden",
    "12_CameraRig",
    "13_Lighting",
    "14_TextLabels",
    "15_RenderHelpers",
]

CAMERA_SPECS = {
    "Camera_WorldHero": ((5.9, -5.7, 4.55), (0.05, 0.65, 1.75), 35),
    "Camera_Arrival": ((0.0, -9.2, 2.25), (0.0, -5.2, 1.55), 38),
    "Camera_ClaimCapsule": ((4.65, -5.4, 3.05), (0.0, 0.0, 1.45), 42),
    "Camera_EvidenceVault": ((-7.15, -1.95, 2.35), (-4.35, 0.0, 1.45), 42),
    "Camera_SupportGraphOrrery": ((3.25, -3.95, 3.95), (0.0, 0.0, 2.95), 62),
    "Camera_NonClaimsWall": ((-0.70, 1.00, 2.05), (-0.70, 3.48, 1.55), 20),
    "Camera_DecayClockRoom": ((8.25, 0.15, 2.25), (4.40, 0.18, 1.55), 38),
    "Camera_ChallengeRange": ((6.6, -7.0, 3.45), (0.25, 0.0, 1.05), 32),
    "Camera_ReplayManifestEngine": ((0.0, -5.8, 4.55), (0.0, 0.0, 0.18), 35),
    "Camera_HumanReviewTribunal": ((7.2, 1.25, 2.35), (4.4, 3.1, 1.2), 48),
    "Camera_LedgerTower": ((-7.4, 1.25, 3.5), (-4.45, 3.45, 2.25), 45),
    "Camera_FrechetAtomGarden": ((-6.35, -4.25, 2.25), (-3.45, -2.15, 0.75), 50),
}

TIMELINE_MARKERS = [
    (0, "000 Arrival Hall", "Camera_Arrival"),
    (60, "060 Claim Capsule", "Camera_ClaimCapsule"),
    (120, "120 Evidence Vault", "Camera_EvidenceVault"),
    (180, "180 Support Graph Orrery", "Camera_SupportGraphOrrery"),
    (240, "240 Non-Claims Wall", "Camera_NonClaimsWall"),
    (300, "300 Decay Clock Room", "Camera_DecayClockRoom"),
    (360, "360 Challenge Range", "Camera_ChallengeRange"),
    (420, "420 Replay Manifest Engine", "Camera_ReplayManifestEngine"),
    (480, "480 Human Review Tribunal", "Camera_HumanReviewTribunal"),
    (540, "540 Ledger Tower", "Camera_LedgerTower"),
    (600, "600 Frechet Atom Garden", "Camera_FrechetAtomGarden"),
    (660, "660 Final Hero", "Camera_WorldHero"),
]

RENDER_TARGETS = [
    ("Camera_WorldHero", "world_v3_hero.png", 660),
    ("Camera_ClaimCapsule", "world_v3_claim_capsule.png", 60),
    ("Camera_NonClaimsWall", "world_v3_nonclaims_wall.png", 240),
    ("Camera_DecayClockRoom", "world_v3_decay_clock.png", 300),
    ("Camera_SupportGraphOrrery", "world_v3_support_graph.png", 180),
    ("Camera_ReplayManifestEngine", "world_v3_replay_engine.png", 420),
]


def read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def load_capsule_data() -> dict[str, Any]:
    envelope = read_json(SOURCE_ARTIFACTS["claim_envelope"])
    audit = read_json(SOURCE_ARTIFACTS["claim_governance_audit"])
    report = read_json(SOURCE_ARTIFACTS["cc_report"])
    manifest = read_json(SOURCE_ARTIFACTS["manifest"])
    decay = read_json(SOURCE_ARTIFACTS["decay_policy"])
    lower = read_json(SOURCE_ARTIFACTS["extremal_lower"])
    upper = read_json(SOURCE_ARTIFACTS["extremal_upper"])
    protocol = read_json(SOURCE_ARTIFACTS["confirmatory_protocol"])
    failure_matrix = read_json(SOURCE_ARTIFACTS["confirmatory_failure_matrix"])

    support_graph = envelope["support_graph"]
    refs = (
        support_graph.get("evidence_refs", [])
        + support_graph.get("scenario_refs", [])
        + support_graph.get("decay_refs", [])
        + support_graph.get("receipt_refs", [])
    )
    ref_by_id = {ref["artifact_id"]: ref for ref in refs}

    return {
        "envelope": envelope,
        "audit": audit,
        "report": report,
        "manifest": manifest,
        "decay": decay,
        "extremal_lower": lower,
        "extremal_upper": upper,
        "protocol": protocol,
        "failure_matrix": failure_matrix,
        "support_edges": support_graph["support_edges"],
        "support_refs_by_id": ref_by_id,
        "source_artifacts": SOURCE_ARTIFACTS,
    }


def set_principled_input(material: bpy.types.Material, name: str, value: Any) -> None:
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf and name in bsdf.inputs:
        bsdf.inputs[name].default_value = value


def make_material(
    name: str,
    base: tuple[float, float, float, float],
    *,
    emission: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
    alpha: float | None = None,
    roughness: float = 0.5,
    metallic: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = base
    set_principled_input(material, "Base Color", base)
    set_principled_input(material, "Roughness", roughness)
    set_principled_input(material, "Metallic", metallic)
    if emission is not None:
        set_principled_input(material, "Emission Color", emission)
        set_principled_input(material, "Emission Strength", emission_strength)
    if alpha is not None:
        set_principled_input(material, "Alpha", alpha)
        material.blend_method = "BLEND"
        material.show_transparent_back = True
    return material


def setup_materials() -> dict[str, bpy.types.Material]:
    return {
        "MAT_World_Dark": make_material("MAT_World_Dark", (0.006, 0.008, 0.012, 1), roughness=0.84),
        "MAT_Capsule_Glass": make_material(
            "MAT_Capsule_Glass",
            (0.35, 0.75, 1.0, 0.24),
            emission=(0.12, 0.45, 0.70, 1),
            emission_strength=0.10,
            alpha=0.24,
            roughness=0.05,
        ),
        "MAT_Evidence_BlueWhite": make_material(
            "MAT_Evidence_BlueWhite",
            (0.68, 0.92, 1.0, 1),
            emission=(0.45, 0.84, 1.0, 1),
            emission_strength=0.80,
            roughness=0.32,
        ),
        "MAT_Diagnostic_Blue": make_material(
            "MAT_Diagnostic_Blue",
            (0.20, 0.46, 0.78, 1),
            emission=(0.12, 0.34, 0.78, 1),
            emission_strength=0.92,
            roughness=0.36,
        ),
        "MAT_Confirmatory_Cyan": make_material(
            "MAT_Confirmatory_Cyan",
            (0.10, 0.88, 1.0, 1),
            emission=(0.06, 0.86, 1.0, 1),
            emission_strength=1.35,
            roughness=0.28,
        ),
        "MAT_Integrity_Pale": make_material(
            "MAT_Integrity_Pale",
            (0.58, 0.68, 0.70, 1),
            emission=(0.30, 0.42, 0.46, 1),
            emission_strength=0.28,
            roughness=0.58,
        ),
        "MAT_Review_Amber": make_material(
            "MAT_Review_Amber",
            (1.0, 0.56, 0.18, 1),
            emission=(1.0, 0.42, 0.10, 1),
            emission_strength=1.10,
            roughness=0.38,
        ),
        "MAT_Invalidation_Red": make_material(
            "MAT_Invalidation_Red",
            (0.95, 0.10, 0.08, 1),
            emission=(0.95, 0.04, 0.03, 1),
            emission_strength=1.20,
            roughness=0.42,
        ),
        "MAT_NonClaims_Stone": make_material("MAT_NonClaims_Stone", (0.018, 0.020, 0.025, 1), roughness=0.78),
        "MAT_Replay_Ledger": make_material("MAT_Replay_Ledger", (0.010, 0.014, 0.018, 1), roughness=0.68),
        "MAT_Future_LedgerTower": make_material(
            "MAT_Future_LedgerTower",
            (0.055, 0.075, 0.092, 1),
            emission=(0.05, 0.16, 0.22, 1),
            emission_strength=0.20,
            roughness=0.5,
        ),
        "MAT_Text_Primary": make_material(
            "MAT_Text_Primary",
            (0.86, 0.94, 1.0, 1),
            emission=(0.74, 0.90, 1.0, 1),
            emission_strength=0.88,
            roughness=0.4,
        ),
        "MAT_Text_Secondary": make_material(
            "MAT_Text_Secondary",
            (0.42, 0.62, 0.72, 1),
            emission=(0.22, 0.42, 0.54, 1),
            emission_strength=0.38,
            roughness=0.5,
        ),
        "MAT_Text_Amber": make_material(
            "MAT_Text_Amber",
            (1.0, 0.62, 0.24, 1),
            emission=(1.0, 0.44, 0.12, 1),
            emission_strength=0.94,
            roughness=0.42,
        ),
        "MAT_Text_Red": make_material(
            "MAT_Text_Red",
            (1.0, 0.16, 0.12, 1),
            emission=(1.0, 0.06, 0.04, 1),
            emission_strength=1.05,
            roughness=0.44,
        ),
        "MAT_Clear_Panel": make_material(
            "MAT_Clear_Panel",
            (0.10, 0.18, 0.22, 0.36),
            emission=(0.05, 0.16, 0.22, 1),
            emission_strength=0.10,
            alpha=0.36,
            roughness=0.48,
        ),
    }


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for item in list(block):
            block.remove(item)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def new_collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    if parent is None:
        bpy.context.scene.collection.children.link(collection)
    else:
        parent.children.link(collection)
    return collection


def setup_collections() -> dict[str, bpy.types.Collection]:
    root = new_collection("00_WorldRoot")
    collections = {"00_WorldRoot": root}
    for name in COLLECTION_NAMES[1:]:
        collections[name] = new_collection(name, root)
    return collections


def link_to(collection: bpy.types.Collection, obj: bpy.types.Object) -> bpy.types.Object:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def tag(obj: bpy.types.Object, **properties: Any) -> bpy.types.Object:
    for key, value in properties.items():
        if isinstance(value, (dict, list)):
            obj[key] = json.dumps(value, sort_keys=True)
        elif value is None:
            obj[key] = "null"
        else:
            obj[key] = value
    return obj


def look_at(obj: bpy.types.Object, target: Vector, *, track: str = "-Z", up: str = "Y") -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat(track, up).to_euler()


def add_cube(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float] | Vector,
    dimensions: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bevel: float = 0.0,
    bevel_segments: int = 3,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel > 0:
        mod = obj.modifiers.new("semantic bevel", "BEVEL")
        mod.width = bevel
        mod.segments = bevel_segments
        obj.modifiers.new("weighted normals", "WEIGHTED_NORMAL")
    return link_to(collection, obj)


def add_cylinder(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float] | Vector,
    radius: float,
    depth: float,
    material: bpy.types.Material,
    *,
    vertices: int = 96,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return link_to(collection, obj)


def add_sphere(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float] | Vector,
    radius: float,
    material: bpy.types.Material,
    *,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    segments: int = 32,
    ring_count: int = 16,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=ring_count,
        radius=radius,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return link_to(collection, obj)


def add_torus(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float] | Vector,
    major_radius: float,
    minor_radius: float,
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    major_segments: int = 144,
    minor_segments: int = 8,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_segments=major_segments,
        minor_segments=minor_segments,
        major_radius=major_radius,
        minor_radius=minor_radius,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return link_to(collection, obj)


def add_curve(
    collection: bpy.types.Collection,
    name: str,
    points: list[Vector],
    material: bpy.types.Material,
    *,
    bevel_depth: float = 0.006,
    resolution: int = 2,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = resolution
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, co in zip(spline.points, points):
        point.co = (co.x, co.y, co.z, 1.0)
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    return obj


def arc_points(
    center: Vector,
    radius: float,
    start_deg: float,
    end_deg: float,
    *,
    plane: str = "XY",
    fixed: float | None = None,
    steps: int = 80,
) -> list[Vector]:
    points = []
    for step in range(steps + 1):
        angle = math.radians(start_deg + (end_deg - start_deg) * step / steps)
        c = math.cos(angle) * radius
        s = math.sin(angle) * radius
        if plane == "XY":
            points.append(Vector((center.x + c, center.y + s, center.z if fixed is None else fixed)))
        elif plane == "XZ":
            points.append(Vector((center.x + c, center.y if fixed is None else fixed, center.z + s)))
        elif plane == "YZ":
            points.append(Vector((center.x if fixed is None else fixed, center.y + c, center.z + s)))
        else:
            raise ValueError(f"unknown arc plane: {plane}")
    return points


def add_text(
    collection: bpy.types.Collection,
    name: str,
    body: str,
    location: tuple[float, float, float] | Vector,
    material: bpy.types.Material,
    *,
    size: float = 0.16,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    align_x: str = "CENTER",
    align_y: str = "CENTER",
    extrude: float = 0.001,
    line_spacing: float = 0.88,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = body
    curve.align_x = align_x
    curve.align_y = align_y
    curve.size = size
    curve.extrude = extrude
    curve.space_line = line_spacing
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    obj.location = location
    obj.rotation_euler = rotation
    collection.objects.link(obj)
    return obj


def add_facing_text(
    collection: bpy.types.Collection,
    name: str,
    body: str,
    location: tuple[float, float, float] | Vector,
    material: bpy.types.Material,
    camera_location: Vector,
    *,
    size: float = 0.16,
    align_x: str = "CENTER",
    align_y: str = "CENTER",
    extrude: float = 0.001,
    line_spacing: float = 0.88,
) -> bpy.types.Object:
    obj = add_text(
        collection,
        name,
        body,
        location,
        material,
        size=size,
        align_x=align_x,
        align_y=align_y,
        extrude=extrude,
        line_spacing=line_spacing,
    )
    look_at(obj, camera_location, track="Z", up="Y")
    return obj


def short_hash(value: str | None, length: int = 10) -> str:
    return "none" if not value else value[:length]


def display_filename_from_artifact_id(artifact_id: str) -> str:
    if ":" in artifact_id:
        tail = artifact_id.split(":")[-1]
        if "." in tail:
            return tail
    return artifact_id


def material_for_strength(materials: dict[str, bpy.types.Material], strength: str, relation: str) -> bpy.types.Material:
    if relation == "invalidates":
        return materials["MAT_Invalidation_Red"]
    if relation == "requires_review":
        return materials["MAT_Review_Amber"]
    if strength == "confirmatory":
        return materials["MAT_Confirmatory_Cyan"]
    if strength == "integrity_only":
        return materials["MAT_Integrity_Pale"]
    return materials["MAT_Diagnostic_Blue"]


def setup_render() -> None:
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = FRAME_END
    scene.frame_set(FRAME_END)
    scene.render.fps = FPS
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.film_transparent = False

    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue

    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 48
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = True
            scene.eevee.bloom_intensity = 0.025
            scene.eevee.bloom_radius = 3.2
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
            scene.eevee.gtao_distance = 4
            scene.eevee.gtao_factor = 1.2

    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "High Contrast", "None"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    scene.view_settings.exposure = -0.45
    scene.view_settings.gamma = 1.0

    world = scene.world or bpy.data.worlds.new("ClaimObservatoryWorldV3")
    scene.world = world
    world.color = (0.002, 0.003, 0.006)


def build_world_root(collections: dict[str, bpy.types.Collection], materials: dict[str, bpy.types.Material]) -> None:
    root = collections["00_WorldRoot"]
    helpers = collections["15_RenderHelpers"]
    add_cube(root, "WorldRoot_SharedDarkFloor", (0, -0.25, -0.045), (11.8, 10.6, 0.09), materials["MAT_World_Dark"], bevel=0.02)

    chamber_pads = [
        ("ArrivalHall_Pad", (0, -5.25, 0.01), (4.7, 2.25, 0.035)),
        ("ClaimCapsuleChamber_Pad", (0, 0, 0.02), (3.4, 3.2, 0.04)),
        ("EvidenceVault_Pad", (-4.25, 0, 0.02), (2.3, 3.05, 0.04)),
        ("DecayClockRoom_Pad", (4.35, 0.2, 0.02), (2.3, 3.05, 0.04)),
        ("NonClaimsWall_Pad", (0, 3.15, 0.02), (6.2, 1.55, 0.04)),
        ("HumanReviewTribunal_Pad", (4.55, 3.25, 0.02), (2.3, 1.65, 0.04)),
        ("LedgerTower_Pad", (-4.45, 3.2, 0.02), (2.1, 1.7, 0.04)),
        ("FrechetAtomGarden_Pad", (-3.45, -2.75, 0.02), (2.3, 1.55, 0.04)),
    ]
    for name, loc, dims in chamber_pads:
        obj = add_cube(helpers, name, loc, dims, materials["MAT_Replay_Ledger"], bevel=0.018)
        tag(obj, semantic_source="visual_identity/claim_observatory/WORLD_BIBLE_V3.md", chamber=name.removesuffix("_Pad"))

    corridor_specs = [
        ("Corridor_Arrival_To_Capsule", (0, -2.7, 0.04), (1.35, 3.0, 0.035)),
        ("Corridor_Capsule_To_EvidenceVault", (-2.1, 0, 0.04), (2.0, 0.55, 0.035)),
        ("Corridor_Capsule_To_DecayClock", (2.1, 0, 0.04), (2.0, 0.55, 0.035)),
        ("Corridor_Capsule_To_NonClaimsWall", (0, 1.75, 0.04), (1.05, 2.3, 0.035)),
        ("Corridor_To_ReviewAndLedger", (0, 3.05, 0.045), (6.4, 0.34, 0.035)),
    ]
    for name, loc, dims in corridor_specs:
        add_cube(helpers, name, loc, dims, materials["MAT_Integrity_Pale"], bevel=0.012)


def build_arrival_hall(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    arrival = collections["01_ArrivalHall"]
    text = collections["14_TextLabels"]
    camera_hint = Vector(CAMERA_SPECS["Camera_Arrival"][0])
    add_cube(arrival, "ArrivalHall_LeftQuietWall", (-2.45, -5.35, 1.1), (0.12, 2.35, 2.2), materials["MAT_NonClaims_Stone"], bevel=0.015)
    add_cube(arrival, "ArrivalHall_RightQuietWall", (2.45, -5.35, 1.1), (0.12, 2.35, 2.2), materials["MAT_NonClaims_Stone"], bevel=0.015)
    add_cube(arrival, "ArrivalHall_BackThesisPanel", (0, -6.42, 1.35), (4.55, 0.10, 2.25), materials["MAT_NonClaims_Stone"], bevel=0.02)
    add_facing_text(
        text,
        "ArrivalHall_ThesisText",
        "Not safety scores.\nEvidence-bound claims.",
        (-1.78, -6.50, 1.88),
        materials["MAT_Text_Primary"],
        camera_hint,
        size=0.26,
        align_x="LEFT",
        line_spacing=0.92,
    )
    add_facing_text(
        text,
        "ArrivalHall_Subtext",
        "A claim is a bounded artifact with receipts, non-claims,\ndecay, challenge surfaces, and replay.",
        (-1.78, -6.50, 1.08),
        materials["MAT_Text_Secondary"],
        camera_hint,
        size=0.085,
        align_x="LEFT",
        line_spacing=0.92,
    )
    label = add_facing_text(
        text,
        "ArrivalHall_SourceReportHash",
        f"source_report_hash: {short_hash(data['envelope']['identity']['source_report_hash'], 12)}",
        (1.15, -6.50, 0.65),
        materials["MAT_Integrity_Pale"],
        camera_hint,
        size=0.056,
        align_x="LEFT",
    )
    tag(label, semantic_source=SOURCE_ARTIFACTS["claim_envelope"], repo_artifact_path=SOURCE_ARTIFACTS["claim_envelope"])


def build_claim_capsule(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    chamber = collections["02_ClaimCapsuleChamber"]
    text = collections["14_TextLabels"]
    audit = data["audit"]
    envelope = data["envelope"]
    report = data["report"]
    manifest = data["manifest"]
    camera_hint = Vector(CAMERA_SPECS["Camera_ClaimCapsule"][0])

    body = add_cylinder(chamber, "ClaimCapsule_BoundedArtifactBody", CAPSULE_CENTER, 0.84, 2.35, materials["MAT_Capsule_Glass"], vertices=144)
    tag(
        body,
        semantic_source=SOURCE_ARTIFACTS["claim_envelope"],
        repo_artifact_path=SOURCE_ARTIFACTS["claim_envelope"],
        schema=envelope["schema"],
        claim_meaning=envelope["proposition"]["statement"],
        does_not_claim="deployment safety",
    )
    add_sphere(chamber, "ClaimCapsule_TopGlassCap", CAPSULE_CENTER + Vector((0, 0, 1.18)), 0.84, materials["MAT_Capsule_Glass"], scale=(1, 1, 0.36), segments=64, ring_count=16)
    add_sphere(chamber, "ClaimCapsule_BottomGlassCap", CAPSULE_CENTER + Vector((0, 0, -1.18)), 0.84, materials["MAT_Capsule_Glass"], scale=(1, 1, 0.36), segments=64, ring_count=16)
    for idx, z_offset in enumerate((-1.15, -0.58, 0.0, 0.58, 1.15), 1):
        add_torus(chamber, f"ClaimCapsule_StrongGlassRim_{idx:02d}", CAPSULE_CENTER + Vector((0, 0, z_offset)), 0.845, 0.0075, materials["MAT_Evidence_BlueWhite"], major_segments=160)

    plates = [
        ("cc_report.json", report["schema_version"], SOURCE_ARTIFACTS["cc_report"], -0.48, "cc.report.v0.3.1"),
        ("claim_governance_audit.json", audit["schema"], SOURCE_ARTIFACTS["claim_governance_audit"], -0.16, "cc/claim-governance-audit.v1"),
        ("claim_envelope.json", envelope["schema"], SOURCE_ARTIFACTS["claim_envelope"], 0.16, "cc.claim_envelope.v1"),
        ("manifest.expected.json", manifest["schema_version"], SOURCE_ARTIFACTS["manifest"], 0.48, "cc.claim_governance_capsule_manifest.v1"),
    ]
    for idx, (filename, schema, source, z_offset, label) in enumerate(plates):
        z = CAPSULE_CENTER.z + z_offset
        plate = add_cube(
            chamber,
            f"ClaimCapsule_InternalEvidencePlate_{idx+1:02d}_{filename.replace('.', '_')}",
            (0.0, -0.055 - idx * 0.012, z),
            (1.24, 0.018, 0.22),
            materials["MAT_Clear_Panel"],
            bevel=0.01,
        )
        tag(plate, semantic_source=source, repo_artifact_path=source, schema=schema, does_not_claim="deployment safety")
        add_facing_text(
            text,
            f"ClaimCapsule_InternalEvidencePlateLabel_{idx+1:02d}",
            f"{filename}\n{label}",
            (-0.54, -0.085 - idx * 0.012, z + 0.02),
            materials["MAT_Text_Primary"],
            camera_hint,
            size=0.044,
            align_x="LEFT",
            line_spacing=0.86,
        )

    add_facing_text(
        text,
        "ClaimCapsule_Status_PASSUnderVerifierRules",
        "PASS under verifier rules",
        (-0.80, -1.02, 0.46),
        materials["MAT_Confirmatory_Cyan"],
        camera_hint,
        size=0.105,
        align_x="LEFT",
    )
    add_facing_text(
        text,
        "ClaimCapsule_Caveat_NotDeploymentSafetyProof",
        "Not a deployment-safety proof",
        (-0.80, -1.02, 0.26),
        materials["MAT_Text_Amber"],
        camera_hint,
        size=0.072,
        align_x="LEFT",
    )
    support_summary = audit["envelope_support"]
    add_facing_text(
        text,
        "ClaimCapsule_SmallStatusFields",
        f"claim_level: {audit['allowed_claim_level']}\n"
        f"freshness: {envelope['governance_state']['freshness_status']}\n"
        f"support_edges: {support_summary['support_edge_count']}",
        (0.34, -1.04, 0.34),
        materials["MAT_Text_Secondary"],
        camera_hint,
        size=0.052,
        align_x="LEFT",
        line_spacing=0.82,
    )


def build_evidence_vault(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    vault = collections["03_EvidenceVault"]
    text = collections["14_TextLabels"]
    manifest = data["manifest"]
    camera_hint = Vector(CAMERA_SPECS["Camera_EvidenceVault"][0])

    add_cube(vault, "EvidenceVault_BackWall", (-4.65, 1.38, 1.25), (2.6, 0.12, 2.5), materials["MAT_NonClaims_Stone"], bevel=0.02)
    add_cube(vault, "EvidenceVault_LeftRack", (-5.50, 0.05, 1.05), (0.10, 2.35, 2.0), materials["MAT_NonClaims_Stone"], bevel=0.012)
    add_cube(vault, "EvidenceVault_RightRack", (-3.50, 0.05, 1.05), (0.10, 2.35, 2.0), materials["MAT_NonClaims_Stone"], bevel=0.012)
    add_facing_text(text, "EvidenceVault_Title", "Evidence Vault", (-5.45, -1.33, 2.2), materials["MAT_Text_Primary"], camera_hint, size=0.14, align_x="LEFT")
    add_facing_text(text, "EvidenceVault_Principle", "Evidence is not generic.\nEvidence has roles.", (-5.45, -1.33, 1.92), materials["MAT_Text_Secondary"], camera_hint, size=0.065, align_x="LEFT")

    files = manifest["files"]
    role_rank = {
        "measurement_evidence": 0,
        "calibration_evidence": 1,
        "confirmatory_failure_matrix": 2,
        "confirmatory_protocol": 3,
        "claim_decay": 4,
        "extremal_scenario": 5,
        "cc_report": 6,
        "claim_envelope": 7,
        "claim_governance_audit": 8,
        "audit_log": 9,
    }
    sorted_files = sorted(files, key=lambda item: (role_rank.get(item["role"], 99), item["filename"]))
    positions = []
    for row in range(4):
        for col in range(3):
            positions.append((-5.15 + col * 0.65, -0.72 + row * 0.55, 0.58 + row * 0.30))

    for idx, file_info in enumerate(sorted_files[:11]):
        loc = Vector(positions[idx])
        role = file_info["role"]
        material = materials["MAT_Confirmatory_Cyan"] if "confirmatory" in role else materials["MAT_Diagnostic_Blue"]
        if role in {"audit_log", "claim_envelope", "claim_governance_audit", "cc_report"}:
            material = materials["MAT_Evidence_BlueWhite"]
        if role == "claim_decay":
            material = materials["MAT_Review_Amber"]
        if role == "extremal_scenario":
            material = materials["MAT_Diagnostic_Blue"]
        tablet = add_cube(
            vault,
            f"EvidenceVault_Tablet_{idx+1:02d}_{file_info['filename'].replace('.', '_')}",
            loc,
            (0.54, 0.035, 0.36),
            material,
            rotation=(0.0, 0.0, math.radians(2)),
            bevel=0.014,
        )
        tag(
            tablet,
            semantic_source=SOURCE_ARTIFACTS["manifest"],
            repo_artifact_path=f"examples/claim_governance_capsule/expected/{file_info['filename']}",
            role=role,
            sha256=file_info["sha256"],
        )
        add_facing_text(
            text,
            f"EvidenceVault_TabletLabel_{idx+1:02d}",
            f"{file_info['filename']}\n{role}\nsha256:{short_hash(file_info['sha256'], 8)}  bytes:{file_info['bytes']}",
            loc + Vector((-0.24, -0.07, 0.02)),
            materials["MAT_Text_Primary"],
            camera_hint,
            size=0.030,
            align_x="LEFT",
            line_spacing=0.78,
        )

    receipt = add_cube(vault, "EvidenceVault_ReceiptIntegrityTablet", (-4.55, -1.08, 0.55), (1.45, 0.040, 0.32), materials["MAT_Integrity_Pale"], bevel=0.012)
    tag(
        receipt,
        semantic_source=SOURCE_ARTIFACTS["manifest"],
        repo_artifact_path=SOURCE_ARTIFACTS["manifest"],
        role="receipt_integrity",
        sha256=manifest["report_receipt_sha256"],
        does_not_claim="statistical validity or deployment safety",
    )
    add_facing_text(
        text,
        "EvidenceVault_ReceiptIntegrityLabel",
        f"Receipt integrity tablet\nsha256:{short_hash(manifest['report_receipt_sha256'], 12)}\nintegrity only != safety proof",
        (-5.17, -1.16, 0.58),
        materials["MAT_Integrity_Pale"],
        camera_hint,
        size=0.042,
        align_x="LEFT",
        line_spacing=0.82,
    )


def build_support_graph_orrery(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    graph = collections["04_SupportGraphOrrery"]
    text = collections["14_TextLabels"]
    edges = data["support_edges"]
    ref_by_id = data["support_refs_by_id"]
    camera_hint = Vector(CAMERA_SPECS["Camera_SupportGraphOrrery"][0])

    add_torus(graph, "SupportGraphOrrery_InnerClaimFragmentRing", CAPSULE_CENTER + Vector((0, 0, 1.35)), 1.10, 0.004, materials["MAT_Integrity_Pale"], rotation=(math.radians(83), 0, math.radians(12)), major_segments=160)
    add_torus(graph, "SupportGraphOrrery_OuterEvidenceArtifactRing", CAPSULE_CENTER + Vector((0, 0, 1.35)), 2.15, 0.004, materials["MAT_Evidence_BlueWhite"], rotation=(math.radians(83), 0, math.radians(12)), major_segments=160)

    fragments = sorted({edge["target_claim_fragment"] for edge in edges})
    sources = [edge["source_artifact_id"] for edge in edges]
    fragment_positions: dict[str, Vector] = {}
    source_positions: dict[str, Vector] = {}
    center = CAPSULE_CENTER + Vector((0, 0, 1.28))

    for idx, fragment in enumerate(fragments):
        angle = math.radians(90 + idx * 360 / len(fragments))
        loc = center + Vector((math.cos(angle) * 0.98, math.sin(angle) * 0.26, math.sin(angle) * 0.58))
        fragment_positions[fragment] = loc
        node = add_sphere(graph, f"SupportGraphOrrery_ClaimFragment_{idx+1:02d}", loc, 0.055, materials["MAT_Text_Primary"], segments=20, ring_count=10)
        tag(node, semantic_source=SOURCE_ARTIFACTS["claim_envelope"], target_claim_fragment=fragment)
        add_facing_text(text, f"SupportGraphOrrery_ClaimFragmentLabel_{idx+1:02d}", fragment.replace("claim.", ""), loc + Vector((0, 0, 0.13)), materials["MAT_Text_Primary"], camera_hint, size=0.036)

    for idx, source in enumerate(sources):
        angle = math.radians(74 + idx * 360 / len(sources))
        loc = center + Vector((math.cos(angle) * 2.05, math.sin(angle) * 0.48, math.sin(angle) * 1.03))
        source_positions[source] = loc
        ref = ref_by_id.get(source, {})
        material = material_for_strength(materials, edges[idx]["strength"], edges[idx]["relation"])
        node = add_sphere(graph, f"SupportGraphOrrery_EvidenceArtifact_{idx+1:02d}", loc, 0.070, material, segments=20, ring_count=10)
        tag(node, semantic_source=SOURCE_ARTIFACTS["claim_envelope"], artifact_id=source, role=ref.get("role", "unknown"), sha256=ref.get("sha256"))
        add_facing_text(text, f"SupportGraphOrrery_EvidenceArtifactLabel_{idx+1:02d}", display_filename_from_artifact_id(source), loc + Vector((0, 0, 0.14)), material, camera_hint, size=0.034)

    for idx, edge in enumerate(edges):
        start = source_positions[edge["source_artifact_id"]]
        end = fragment_positions[edge["target_claim_fragment"]]
        midpoint = start.lerp(end, 0.52) + Vector((0.0, 0.0, 0.11 + 0.02 * (idx % 3)))
        material = material_for_strength(materials, edge["strength"], edge["relation"])
        width = 0.009
        if edge["strength"] == "integrity_only":
            width = 0.004
        elif edge["strength"] == "confirmatory":
            width = 0.012
        beam = add_curve(graph, f"SupportGraphOrrery_Edge_{idx+1:02d}_{edge['relation']}", [start, midpoint, end], material, bevel_depth=width)
        tag(
            beam,
            semantic_source=SOURCE_ARTIFACTS["claim_envelope"],
            relation=edge["relation"],
            strength=edge["strength"],
            source_artifact_id=edge["source_artifact_id"],
            target_claim_fragment=edge["target_claim_fragment"],
        )
        if idx % 2 == 0 or edge["strength"] == "integrity_only":
            add_facing_text(text, f"SupportGraphOrrery_EdgeLabel_{idx+1:02d}", edge["relation"], midpoint + Vector((0, 0, 0.08)), material, camera_hint, size=0.034)

    add_facing_text(text, "SupportGraphOrrery_Title", "Support is typed.", (1.30, -1.35, 3.82), materials["MAT_Text_Primary"], camera_hint, size=0.095, align_x="LEFT")
    add_facing_text(text, "SupportGraphOrrery_Legend", "Role determines what evidence may say.\nconfirmatory / diagnostic / integrity only", (1.30, -1.35, 3.60), materials["MAT_Text_Secondary"], camera_hint, size=0.052, align_x="LEFT", line_spacing=0.82)


def build_non_claims_wall(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    wall = collections["05_NonClaimsWall"]
    text = collections["14_TextLabels"]
    camera_hint = Vector(CAMERA_SPECS["Camera_NonClaimsWall"][0])
    manifest = data["manifest"]

    monolith = add_cube(wall, "NonClaimsWall_MonumentalSemanticFirewall", (0.0, 3.55, 1.55), (6.2, 0.18, 2.65), materials["MAT_NonClaims_Stone"], bevel=0.028, bevel_segments=5)
    tag(monolith, semantic_source=SOURCE_ARTIFACTS["claim_governance_audit"], repo_artifact_path=SOURCE_ARTIFACTS["claim_governance_audit"], does_not_claim="deployment safety")
    for idx, x in enumerate([-2.72, -2.05, -1.36, -0.68, 0.0, 0.68, 1.36, 2.05, 2.72], 1):
        add_curve(wall, f"NonClaimsWall_VerticalBoundaryRib_{idx:02d}", [Vector((x, 3.43, 0.38)), Vector((x, 3.43, 2.76))], materials["MAT_Integrity_Pale"], bevel_depth=0.003)
    add_curve(wall, "NonClaimsWall_ProtectiveOverreachArc", arc_points(Vector((0, 3.42, 1.52)), 3.0, 198, 342, plane="XZ", fixed=3.41), materials["MAT_Text_Amber"], bevel_depth=0.012)

    wall_title_body = "THIS CLAIM DOES NOT SAY"
    wall_rotation = (math.radians(90), 0.0, 0.0)
    wall_text_y = 3.43
    add_text(text, "NonClaimsWall_Title", wall_title_body.replace(" CLAIM ", " CLAIM\n"), (-2.22, wall_text_y, 2.42), materials["MAT_Text_Amber"], size=0.130, rotation=wall_rotation, align_x="LEFT", line_spacing=0.86)
    non_claim_labels = [
        "deployment safety",
        "external validity",
        "release approval",
        "future freshness",
        "statistical truth beyond scope",
    ]
    add_text(text, "NonClaimsWall_MinimumBoundaryList", "\n".join(non_claim_labels), (-2.22, wall_text_y, 1.62), materials["MAT_Text_Primary"], size=0.072, rotation=wall_rotation, align_x="LEFT", line_spacing=0.92)
    caveat = manifest["pass_caveat"].replace("; ", ";\n")
    add_text(text, "NonClaimsWall_ExactPassCaveat", caveat, (0.05, wall_text_y, 1.30), materials["MAT_Text_Secondary"], size=0.052, rotation=wall_rotation, align_x="LEFT", line_spacing=0.86)
    add_text(text, "NonClaimsWall_PhysicalPurpose", "Non-claims attached.\nThe wall prevents claim overreach.", (0.05, wall_text_y, 2.30), materials["MAT_Confirmatory_Cyan"], size=0.062, rotation=wall_rotation, align_x="LEFT", line_spacing=0.88)


def build_decay_clock_room(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    decay_room = collections["06_DecayClockRoom"]
    text = collections["14_TextLabels"]
    audit = data["audit"]
    decay = data["decay"]
    camera_hint = Vector(CAMERA_SPECS["Camera_DecayClockRoom"][0])
    center = Vector((4.36, 0.20, 1.55))

    add_cube(decay_room, "DecayClockRoom_BackPlate", (4.72, 1.34, 1.42), (2.28, 0.12, 2.45), materials["MAT_NonClaims_Stone"], bevel=0.02)
    add_curve(decay_room, "DecayClockRoom_FreshArc", arc_points(center, 1.02, 42, 158, plane="YZ", fixed=center.x), materials["MAT_Confirmatory_Cyan"], bevel_depth=0.020)
    add_curve(decay_room, "DecayClockRoom_DegradedArc", arc_points(center, 1.02, 158, 286, plane="YZ", fixed=center.x), materials["MAT_Review_Amber"], bevel_depth=0.018)
    add_curve(decay_room, "DecayClockRoom_ExpiredArc", arc_points(center, 1.02, 286, 402, plane="YZ", fixed=center.x), materials["MAT_Invalidation_Red"], bevel_depth=0.016)
    add_curve(decay_room, "DecayClockRoom_OuterQuietRing", arc_points(center, 1.12, 0, 360, plane="YZ", fixed=center.x), materials["MAT_Integrity_Pale"], bevel_depth=0.004)

    status_angle_by_name = {"fresh": 82, "degraded": 218, "expired": 330}
    status = audit["decay"]["status"]
    angle = math.radians(status_angle_by_name.get(status, 82))
    marker = Vector((center.x - 0.03, center.y + math.cos(angle) * 1.02, center.z + math.sin(angle) * 1.02))
    marker_obj = add_sphere(decay_room, f"DecayClockRoom_CurrentStatusMarker_{status}", marker, 0.075, materials["MAT_Text_Primary"], segments=24, ring_count=12)
    tag(marker_obj, semantic_source=SOURCE_ARTIFACTS["claim_governance_audit"], repo_artifact_path=SOURCE_ARTIFACTS["decay_policy"], status=status)
    add_curve(decay_room, "DecayClockRoom_StatusNeedle", [center, marker], materials["MAT_Text_Primary"], bevel_depth=0.005)

    add_facing_text(text, "DecayClockRoom_Title", "claims are mortal", (3.52, -0.98, 2.72), materials["MAT_Text_Primary"], camera_hint, size=0.115, align_x="LEFT")
    add_facing_text(text, "DecayClockRoom_StatusLabels", "Fresh\nDegraded\nExpired", (5.18, -0.78, 2.34), materials["MAT_Text_Secondary"], camera_hint, size=0.065, align_x="LEFT", line_spacing=1.1)
    add_facing_text(
        text,
        "DecayClockRoom_RealDecayFields",
        f"status: {status}\n"
        f"issued_at: {decay['issued_at']}\n"
        f"evaluated_at: {audit['evaluated_at']}\n"
        f"degraded_after_days: {decay['policy']['degraded_after_days']}\n"
        f"expires_after_days: {decay['policy']['expires_after_days']}",
        (3.48, -0.98, 0.58),
        materials["MAT_Text_Secondary"],
        camera_hint,
        size=0.048,
        align_x="LEFT",
        line_spacing=0.82,
    )


def build_challenge_range(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    challenge = collections["07_ChallengeRange"]
    text = collections["14_TextLabels"]
    camera_hint = Vector(CAMERA_SPECS["Camera_ChallengeRange"][0])
    add_torus(challenge, "ChallengeRange_BoundaryRing_ChallengeSurface", (0, 0, 0.13), 5.10, 0.010, materials["MAT_Review_Amber"], major_segments=192, minor_segments=6)
    add_facing_text(text, "ChallengeRange_RingLabel", "challenge surface", (2.70, -4.25, 0.52), materials["MAT_Text_Amber"], camera_hint, size=0.060)

    probes = [
        ("hash mismatch", "FAIL"),
        ("missing non-claim", "NEEDS_REVIEW"),
        ("expired decay", "FAIL"),
        ("unknown role", "NEEDS_REVIEW"),
        ("schema null", "review trigger"),
        ("exploratory leakage", "semantic continuity failure"),
        ("infeasible scenario", "FAIL"),
        ("confirmatory protocol failure", "NEEDS_REVIEW"),
    ]
    for idx, (probe, expected) in enumerate(probes):
        angle = math.radians(18 + idx * 360 / len(probes))
        loc = Vector((math.cos(angle) * 5.05, math.sin(angle) * 5.05, 0.48 + 0.08 * (idx % 2)))
        material = materials["MAT_Invalidation_Red"] if expected == "FAIL" else materials["MAT_Review_Amber"]
        instrument = add_cube(challenge, f"ChallengeRange_Probe_{idx+1:02d}_{probe.replace(' ', '_')}", loc, (0.20, 0.20, 0.44), material, rotation=(0, 0, angle), bevel=0.015)
        tag(instrument, semantic_source="visual_identity/claim_observatory/WORLD_BIBLE_V3.md", challenge_probe=probe, expected_result=expected)
        add_curve(challenge, f"ChallengeRange_ProbeBeam_{idx+1:02d}", [loc + Vector((0, 0, 0.26)), CAPSULE_CENTER + Vector((0, 0, 0.05))], material, bevel_depth=0.0028)
        add_facing_text(text, f"ChallengeRange_ProbeLabel_{idx+1:02d}", f"{probe}\n{expected}", loc + Vector((0, 0, 0.48)), material, camera_hint, size=0.042, line_spacing=0.80)


def build_world_hero_labels(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
) -> None:
    text = collections["14_TextLabels"]
    camera_hint = Vector(CAMERA_SPECS["Camera_WorldHero"][0])
    add_facing_text(
        text,
        "WorldHero_Thesis_NotSafetyScores_EvidenceBoundClaims",
        "Not safety scores.\nEvidence-bound claims.",
        (-3.32, -1.28, 3.82),
        materials["MAT_Text_Primary"],
        camera_hint,
        size=0.150,
        align_x="LEFT",
        line_spacing=0.90,
    )
    add_facing_text(
        text,
        "WorldHero_FinalTitle_MeaningSurvivesReplay",
        "Meaning survives replay.",
        (-3.32, -1.28, 3.39),
        materials["MAT_Confirmatory_Cyan"],
        camera_hint,
        size=0.075,
        align_x="LEFT",
    )
    add_facing_text(
        text,
        "WorldHero_Caveat_NotDeploymentSafetyProof",
        "Not a deployment-safety proof",
        (-3.32, -1.28, 3.21),
        materials["MAT_Text_Amber"],
        camera_hint,
        size=0.052,
        align_x="LEFT",
    )


def build_replay_manifest_engine(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    engine = collections["08_ReplayManifestEngine"]
    text = collections["14_TextLabels"]
    manifest = data["manifest"]
    camera_hint = Vector(CAMERA_SPECS["Camera_ReplayManifestEngine"][0])

    floor = add_cylinder(engine, "ReplayManifestEngine_CircularFloor", (0, 0, 0.035), 2.82, 0.07, materials["MAT_Replay_Ledger"], vertices=192)
    tag(floor, semantic_source=SOURCE_ARTIFACTS["manifest"], repo_artifact_path=SOURCE_ARTIFACTS["manifest"], schema=manifest["schema_version"], seed=manifest["seed"])
    for idx, radius in enumerate((0.92, 1.55, 2.18, 2.72), 1):
        add_torus(engine, f"ReplayManifestEngine_ConcentricLedgerRing_{idx:02d}", (0, 0, 0.10), radius, 0.004, materials["MAT_Integrity_Pale"] if idx != 2 else materials["MAT_Evidence_BlueWhite"], major_segments=180, minor_segments=6)

    add_text(engine, "ReplayManifestEngine_FloorTitle", "Replay Manifest Engine", (-0.86, -0.28, 0.12), materials["MAT_Text_Primary"], size=0.125, rotation=(0, 0, 0), align_x="LEFT")
    add_text(engine, "ReplayManifestEngine_MeaningSurvivesReplay", "Meaning survives replay.", (-0.86, -0.55, 0.12), materials["MAT_Confirmatory_Cyan"], size=0.080, rotation=(0, 0, 0), align_x="LEFT")
    add_text(engine, "ReplayManifestEngine_ReproduceCommands", "reproduce.sh\nreproduce.sh --verify-only", (0.70, -0.52, 0.12), materials["MAT_Integrity_Pale"], size=0.060, rotation=(0, 0, 0), align_x="LEFT", line_spacing=0.86)

    files = manifest["files"]
    for idx, file_info in enumerate(files):
        angle = math.radians(90 + idx * 360 / len(files))
        radius = 2.18 if idx % 2 else 2.55
        loc = Vector((math.cos(angle) * radius, math.sin(angle) * radius, 0.125))
        tile = add_cube(
            engine,
            f"ReplayManifestEngine_FileTile_{idx+1:02d}_{file_info['filename'].replace('.', '_')}",
            loc,
            (0.62, 0.035, 0.22),
            materials["MAT_Clear_Panel"],
            rotation=(0, 0, angle + math.pi / 2),
            bevel=0.010,
        )
        tag(tile, semantic_source=SOURCE_ARTIFACTS["manifest"], filename=file_info["filename"], role=file_info["role"], sha256=file_info["sha256"], bytes=file_info["bytes"])
        add_text(
            engine,
            f"ReplayManifestEngine_FileTileLabel_{idx+1:02d}",
            f"{file_info['filename']}\n{file_info['role']}\n{short_hash(file_info['sha256'], 8)}  {file_info['bytes']} B",
            loc + Vector((-0.25, -0.025, 0.12)),
            materials["MAT_Text_Secondary"],
            size=0.027,
            rotation=(0, 0, angle + math.pi / 2),
            align_x="LEFT",
            line_spacing=0.76,
        )

    add_facing_text(
        text,
        "ReplayManifestEngine_CaveatLabel",
        f"manifest: {manifest['schema_version']}\nfixed_now: {manifest['fixed_now']}  seed: {manifest['seed']}",
        (-1.52, -2.18, 0.65),
        materials["MAT_Text_Secondary"],
        camera_hint,
        size=0.046,
        align_x="LEFT",
        line_spacing=0.82,
    )


def build_human_review_tribunal(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    tribunal = collections["09_HumanReviewTribunal"]
    text = collections["14_TextLabels"]
    audit = data["audit"]
    camera_hint = Vector(CAMERA_SPECS["Camera_HumanReviewTribunal"][0])
    add_cube(tribunal, "HumanReviewTribunal_ReviewerChairBase", (4.50, 3.35, 0.42), (0.72, 0.62, 0.16), materials["MAT_NonClaims_Stone"], bevel=0.025)
    add_cube(tribunal, "HumanReviewTribunal_ReviewerChairBack", (4.50, 3.58, 0.96), (0.75, 0.12, 0.90), materials["MAT_NonClaims_Stone"], bevel=0.025)
    add_cube(tribunal, "HumanReviewTribunal_AccountabilityConsole", (4.10, 2.72, 0.72), (1.18, 0.36, 0.30), materials["MAT_World_Dark"], rotation=(0, 0, math.radians(-8)), bevel=0.020)
    add_cube(tribunal, "HumanReviewTribunal_AmberScopedReviewLight", (4.05, 2.52, 0.94), (0.82, 0.026, 0.052), materials["MAT_Review_Amber"], rotation=(0, 0, math.radians(-8)), bevel=0.006)
    add_facing_text(text, "HumanReviewTribunal_Title", "human review", (3.55, 2.20, 1.40), materials["MAT_Text_Amber"], camera_hint, size=0.115, align_x="LEFT")
    add_facing_text(text, "HumanReviewTribunal_Principle", "review authorizes scoped use;\nit does not upgrade evidence", (3.55, 2.20, 1.12), materials["MAT_Text_Secondary"], camera_hint, size=0.060, align_x="LEFT", line_spacing=0.86)
    add_facing_text(text, "HumanReviewTribunal_Requirement", f"required_human_review: {str(audit['required_human_review']).lower()} under current verifier rules", (3.55, 2.20, 0.82), materials["MAT_Text_Primary"], camera_hint, size=0.046, align_x="LEFT")


def build_ledger_tower(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
) -> None:
    tower = collections["10_LedgerTower"]
    text = collections["14_TextLabels"]
    camera_hint = Vector(CAMERA_SPECS["Camera_LedgerTower"][0])
    statuses = ["registered", "reaffirmed", "degraded", "expired", "challenged", "retracted"]
    for idx, status in enumerate(statuses):
        z = 0.38 + idx * 0.36
        material = materials["MAT_Future_LedgerTower"]
        if status in {"degraded", "challenged"}:
            material = materials["MAT_Review_Amber"]
        if status in {"expired", "retracted"}:
            material = materials["MAT_Invalidation_Red"]
        block = add_cube(tower, f"LedgerTower_LifecycleBlock_{idx+1:02d}_{status}", (-4.50, 3.42, z), (0.82, 0.46, 0.22), material, bevel=0.012)
        tag(block, semantic_source="visual_identity/claim_observatory/WORLD_BIBLE_V3.md", future_layer=True, lifecycle_state=status)
        add_facing_text(text, f"LedgerTower_LifecycleLabel_{idx+1:02d}", status, (-4.02, 3.20, z), material, camera_hint, size=0.045, align_x="LEFT")
    add_facing_text(text, "LedgerTower_Title", "claim lifecycle ledger", (-5.20, 2.82, 2.72), materials["MAT_Text_Primary"], camera_hint, size=0.092, align_x="LEFT")
    add_facing_text(text, "LedgerTower_FutureLayer", "future public claim ledger\nobservatory layer, not current capsule truth", (-5.20, 2.82, 2.45), materials["MAT_Text_Secondary"], camera_hint, size=0.050, align_x="LEFT", line_spacing=0.84)


def build_frechet_atom_garden(
    collections: dict[str, bpy.types.Collection],
    materials: dict[str, bpy.types.Material],
    data: dict[str, Any],
) -> None:
    garden = collections["11_FrechetAtomGarden"]
    text = collections["14_TextLabels"]
    lower = data["extremal_lower"]
    upper = data["extremal_upper"]
    camera_hint = Vector(CAMERA_SPECS["Camera_FrechetAtomGarden"][0])
    base = Vector((-3.55, -2.68, 0.35))

    add_facing_text(text, "FrechetAtomGarden_Title", "Frechet Atom Garden", (-4.62, -3.35, 1.55), materials["MAT_Text_Primary"], camera_hint, size=0.090, align_x="LEFT")
    add_facing_text(text, "FrechetAtomGarden_Principle", "finite atoms witness endpoint bounds", (-4.62, -3.35, 1.36), materials["MAT_Text_Secondary"], camera_hint, size=0.048, align_x="LEFT")

    for atom in lower["atom_table"]:
        failures = atom["failures"]
        loc = base + Vector((failures[0] * 0.52, failures[1] * 0.52, failures[2] * 0.38))
        radius = 0.035 + 0.12 * atom["probability"]
        material = materials["MAT_Invalidation_Red"] if atom["event_occurs"] else materials["MAT_Diagnostic_Blue"]
        point = add_sphere(garden, f"FrechetAtomGarden_Atom_{atom['atom_index']:02d}", loc, radius, material, segments=16, ring_count=8)
        tag(point, semantic_source=SOURCE_ARTIFACTS["extremal_lower"], atom_index=atom["atom_index"], probability=atom["probability"], event_occurs=atom["event_occurs"])
        add_facing_text(text, f"FrechetAtomGarden_AtomLabel_{atom['atom_index']:02d}", str(atom["atom_index"]), loc + Vector((0, 0, 0.12)), materials["MAT_Text_Secondary"], camera_hint, size=0.026)

    witnesses = [
        ("lower witness", lower, SOURCE_ARTIFACTS["extremal_lower"], -4.78, materials["MAT_Diagnostic_Blue"]),
        ("upper witness", upper, SOURCE_ARTIFACTS["extremal_upper"], -3.64, materials["MAT_Confirmatory_Cyan"]),
    ]
    for label, scenario, source, x, material in witnesses:
        panel = add_cube(garden, f"FrechetAtomGarden_{label.replace(' ', '_').title()}Panel", (x, -2.10, 0.82), (0.92, 0.04, 0.66), materials["MAT_Clear_Panel"], bevel=0.012)
        tag(panel, semantic_source=source, repo_artifact_path=source, scenario_id=scenario["scenario_id"], kind=scenario["kind"], endpoint=scenario["endpoint"], is_feasible=scenario["feasibility"]["is_feasible"])
        add_facing_text(
            text,
            f"FrechetAtomGarden_{label.replace(' ', '_').title()}Label",
            f"{scenario['kind']}\n{label}\nendpoint: {scenario['endpoint']}\nfeasible: {scenario['feasibility']['is_feasible']}",
            (x - 0.38, -2.16, 0.94),
            material,
            camera_hint,
            size=0.040,
            align_x="LEFT",
            line_spacing=0.82,
        )


def build_lighting(collections: dict[str, bpy.types.Collection]) -> None:
    lighting = collections["13_Lighting"]
    light_specs = [
        ("Lighting_KeySoftbox_World", "AREA", (0.0, -4.4, 6.8), 420, (5.8, 3.0), (0.72, 0.86, 1.0)),
        ("Lighting_Capsule_CyanRim", "POINT", (-2.4, -2.0, 3.5), 160, None, (0.45, 0.82, 1.0)),
        ("Lighting_NonClaims_WallGrazing", "AREA", (0.0, 2.65, 3.4), 170, (5.2, 1.1), (0.85, 0.92, 1.0)),
        ("Lighting_Decay_AmberPractical", "POINT", (4.75, -0.95, 2.45), 120, None, (1.0, 0.55, 0.22)),
        ("Lighting_Challenge_RedEdge", "POINT", (4.9, -3.8, 1.0), 62, None, (1.0, 0.18, 0.14)),
        ("Lighting_Ledger_FutureSpine", "POINT", (-4.8, 2.7, 2.6), 95, None, (0.45, 0.75, 1.0)),
    ]
    for name, light_type, loc, energy, size, color in light_specs:
        data = bpy.data.lights.new(name, type=light_type)
        data.energy = energy
        data.color = color
        if light_type == "AREA" and size:
            data.size = size[0]
            if hasattr(data, "size_y"):
                data.size_y = size[1]
        obj = bpy.data.objects.new(name, data)
        obj.location = loc
        lighting.objects.link(obj)
        if light_type == "AREA":
            look_at(obj, CAPSULE_CENTER)


def build_cameras(collections: dict[str, bpy.types.Collection]) -> dict[str, bpy.types.Object]:
    rig = collections["12_CameraRig"]
    cameras: dict[str, bpy.types.Object] = {}
    for name, (location, target, lens) in CAMERA_SPECS.items():
        camera_data = bpy.data.cameras.new(f"{name}_Data")
        camera = bpy.data.objects.new(name, camera_data)
        camera.location = location
        camera_data.lens = lens
        camera_data.dof.use_dof = True
        camera_data.dof.focus_distance = (Vector(target) - Vector(location)).length
        camera_data.dof.aperture_fstop = 6.3
        look_at(camera, Vector(target))
        rig.objects.link(camera)
        cameras[name] = camera

    scene = bpy.context.scene
    scene.camera = cameras["Camera_WorldHero"]
    for frame, label, camera_name in TIMELINE_MARKERS:
        marker = scene.timeline_markers.new(label, frame=frame)
        marker.camera = cameras[camera_name]
    scene.frame_set(FRAME_END)
    return cameras


def write_visual_manifest(data: dict[str, Any]) -> None:
    audit = data["audit"]
    manifest = data["manifest"]
    envelope = data["envelope"]
    render_entries = [
        {
            "camera": camera_name,
            "path": f"visual_identity/claim_observatory/renders/{filename}",
            "frame": frame,
            "resolution": [1920, 1080],
        }
        for camera_name, filename, frame in RENDER_TARGETS
    ]
    visual_manifest = {
        "schema": "cc.visual_claim_observatory_world.v3",
        "source_artifacts": SOURCE_ARTIFACTS,
        "chambers": [
            {"collection": name, "concept": concept}
            for name, concept in [
                ("01_ArrivalHall", "Core thesis entry"),
                ("02_ClaimCapsuleChamber", "Bounded claim body"),
                ("03_EvidenceVault", "Evidence role taxonomy"),
                ("04_SupportGraphOrrery", "Typed support graph"),
                ("05_NonClaimsWall", "Non-claim boundary"),
                ("06_DecayClockRoom", "Freshness and mortality"),
                ("07_ChallengeRange", "Boundary tests"),
                ("08_ReplayManifestEngine", "Replayable manifest"),
                ("09_HumanReviewTribunal", "Scoped accountable review"),
                ("10_LedgerTower", "Future public claim lifecycle"),
                ("11_FrechetAtomGarden", "Frechet endpoint witnesses"),
            ]
        ],
        "visual_objects": [
            {"name": "ClaimCapsule_BoundedArtifactBody", "maps_to": SOURCE_ARTIFACTS["claim_envelope"]},
            {"name": "EvidenceVault_ReceiptIntegrityTablet", "maps_to": "report_receipt_sha256"},
            {"name": "SupportGraphOrrery_Edge_*", "maps_to": "claim_envelope.support_graph.support_edges"},
            {"name": "NonClaimsWall_MonumentalSemanticFirewall", "maps_to": "claim_governance_audit.non_claims"},
            {"name": "DecayClockRoom_CurrentStatusMarker_fresh", "maps_to": "claim_governance_audit.decay.status"},
            {"name": "ReplayManifestEngine_FileTile_*", "maps_to": "manifest.expected.json.files"},
            {"name": "HumanReviewTribunal_AccountabilityConsole", "maps_to": "claim_governance_audit.required_human_review"},
            {"name": "FrechetAtomGarden_Atom_*", "maps_to": "extremal_lower.atom_table"},
        ],
        "semantic_mappings": {
            "verdict": audit["verdict"],
            "allowed_claim_level": audit["allowed_claim_level"],
            "required_human_review": audit["required_human_review"],
            "evaluated_at": audit["evaluated_at"],
            "decay_status": audit["decay"]["status"],
            "support_edge_count": audit["envelope_support"]["support_edge_count"],
            "relation_counts": audit["envelope_support"]["relation_counts"],
            "integrity_only_edges": audit["envelope_support"]["integrity_only_edges"],
            "freshness_status": envelope["governance_state"]["freshness_status"],
            "pass_caveat": manifest["pass_caveat"],
            "report_receipt_sha256": manifest["report_receipt_sha256"],
            "fixed_now": manifest["fixed_now"],
            "seed": manifest["seed"],
        },
        "forbidden_interpretations": [
            "The render is not evidence.",
            "The render does not claim AI deployment safety.",
            "Receipt integrity is not statistical validity.",
            "A PASS verdict is scoped to verifier rules.",
            "Human review does not upgrade evidence.",
        ],
        "cameras": [{"name": name, "lens": spec[2], "location": list(spec[0]), "target": list(spec[1])} for name, spec in CAMERA_SPECS.items()],
        "renders": render_entries,
    }
    VISUAL_MANIFEST_PATH.write_text(json.dumps(visual_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_scene() -> dict[str, bpy.types.Object]:
    data = load_capsule_data()
    reset_scene()
    setup_render()
    materials = setup_materials()
    collections = setup_collections()

    build_world_root(collections, materials)
    build_replay_manifest_engine(collections, materials, data)
    build_arrival_hall(collections, materials, data)
    build_claim_capsule(collections, materials, data)
    build_evidence_vault(collections, materials, data)
    build_support_graph_orrery(collections, materials, data)
    build_non_claims_wall(collections, materials, data)
    build_decay_clock_room(collections, materials, data)
    build_challenge_range(collections, materials, data)
    build_world_hero_labels(collections, materials)
    build_human_review_tribunal(collections, materials, data)
    build_ledger_tower(collections, materials)
    build_frechet_atom_garden(collections, materials, data)
    build_lighting(collections)
    cameras = build_cameras(collections)
    write_visual_manifest(data)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    return cameras


def render_stills(cameras: dict[str, bpy.types.Object]) -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    for camera_name, filename, frame in RENDER_TARGETS:
        scene.frame_set(frame)
        scene.camera = cameras[camera_name]
        scene.render.filepath = str(RENDER_DIR / filename)
        bpy.ops.render.render(write_still=True)


def render_single_still(cameras: dict[str, bpy.types.Object], requested_camera: str) -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    for camera_name, filename, frame in RENDER_TARGETS:
        if camera_name == requested_camera:
            scene.frame_set(frame)
            scene.camera = cameras[camera_name]
            scene.render.filepath = str(RENDER_DIR / filename)
            bpy.ops.render.render(write_still=True)
            return
    raise SystemExit(f"No render target is configured for camera: {requested_camera}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-still", action="store_true", help="Render the V3 still set after creating the .blend.")
    parser.add_argument("--render-stills", action="store_true", help="Render the V3 still set after creating the .blend.")
    parser.add_argument("--render-camera", help="Render only one configured V3 camera target, for iteration.")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    cameras = build_scene()
    if args.render_camera:
        render_single_still(cameras, args.render_camera)
    elif args.render_still or args.render_stills:
        render_stills(cameras)


if __name__ == "__main__":
    main()
