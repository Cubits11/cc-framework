# SPDX-License-Identifier: MIT
"""Build the cc-framework Claim Observatory cinematic Blender scene.

Run from the repository root:

    /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python visual_identity/claim_observatory/create_claim_observatory_scene.py \
      -- --render-still
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parent
RENDER_DIR = ROOT / "renders"
BLEND_PATH = ROOT / "claim_observatory.blend"
HERO_RENDER_PATH = RENDER_DIR / "claim_observatory_hero.png"

CENTER = Vector((0.0, 0.0, 2.85))
FPS = 24
FRAME_END = 360


def set_principled_input(material: bpy.types.Material, name: str, value) -> None:
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
    roughness: float = 0.45,
    metallic: float = 0.0,
    transmission: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = base

    set_principled_input(material, "Base Color", base)
    set_principled_input(material, "Metallic", metallic)
    set_principled_input(material, "Roughness", roughness)
    if alpha is not None:
        set_principled_input(material, "Alpha", alpha)
        material.blend_method = "BLEND"
        material.use_screen_refraction = True if hasattr(material, "use_screen_refraction") else False
        material.show_transparent_back = True
    if emission is not None:
        set_principled_input(material, "Emission Color", emission)
        set_principled_input(material, "Emission Strength", emission_strength)
    if transmission:
        for socket_name in ("Transmission Weight", "Transmission"):
            set_principled_input(material, socket_name, transmission)
        set_principled_input(material, "Alpha", alpha if alpha is not None else 0.35)
        material.blend_method = "BLEND"
    return material


def make_volume_material(name: str) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    scatter = nodes.new(type="ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (0.18, 0.36, 0.50, 1.0)
    scatter.inputs["Density"].default_value = 0.006
    anisotropy = scatter.inputs.get("Anisotropy")
    if anisotropy:
        anisotropy.default_value = 0.18
    material.node_tree.links.new(scatter.outputs["Volume"], output.inputs["Volume"])
    return material


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.collections,
    ):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def new_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_to(collection: bpy.types.Collection, obj: bpy.types.Object) -> bpy.types.Object:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def look_at(obj: bpy.types.Object, target: Vector, *, track: str = "-Z", up: str = "Y") -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat(track, up).to_euler()


def add_cube(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
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
        mod = obj.modifiers.new("small bevels catch only deliberate light", "BEVEL")
        mod.width = bevel
        mod.segments = bevel_segments
        obj.modifiers.new("weighted normals", "WEIGHTED_NORMAL")
    return link_to(collection, obj)


def add_cylinder(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    *,
    vertices: int = 128,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bevel: bool = False,
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
    if bevel:
        mod = obj.modifiers.new("soft technical bevel", "BEVEL")
        mod.width = 0.02
        mod.segments = 8
        obj.modifiers.new("weighted normals", "WEIGHTED_NORMAL")
    return link_to(collection, obj)


def add_sphere(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
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
    location: tuple[float, float, float],
    major_radius: float,
    minor_radius: float,
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    major_segments: int = 160,
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
    bevel_depth: float = 0.01,
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


def add_dashed_curve(
    collection: bpy.types.Collection,
    name: str,
    start: Vector,
    end: Vector,
    material: bpy.types.Material,
    *,
    segments: int = 11,
    duty: float = 0.54,
    bevel_depth: float = 0.007,
) -> list[bpy.types.Object]:
    pieces: list[bpy.types.Object] = []
    delta = end - start
    for idx in range(segments):
        a = idx / segments
        b = min((idx + duty) / segments, 1.0)
        if idx % 2 == 0:
            pieces.append(
                add_curve(
                    collection,
                    f"{name}_dash_{idx:02d}",
                    [start + delta * a, start + delta * b],
                    material,
                    bevel_depth=bevel_depth,
                )
            )
    return pieces


def arc_points(
    center: Vector,
    radius: float,
    start_deg: float,
    end_deg: float,
    *,
    plane: str = "XZ",
    y: float = 0.0,
    steps: int = 80,
) -> list[Vector]:
    points: list[Vector] = []
    for step in range(steps + 1):
        t = math.radians(start_deg + (end_deg - start_deg) * step / steps)
        if plane == "XZ":
            points.append(Vector((center.x + math.cos(t) * radius, y, center.z + math.sin(t) * radius)))
        elif plane == "XY":
            points.append(Vector((center.x + math.cos(t) * radius, center.y + math.sin(t) * radius, center.z)))
        else:
            points.append(Vector((center.x, center.y + math.cos(t) * radius, center.z + math.sin(t) * radius)))
    return points


def add_text(
    collection: bpy.types.Collection,
    name: str,
    body: str,
    location: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    size: float = 0.18,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    align_x: str = "CENTER",
    align_y: str = "CENTER",
    extrude: float = 0.002,
    line_spacing: float = 0.92,
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
    location: tuple[float, float, float],
    material: bpy.types.Material,
    camera_location: Vector,
    *,
    size: float = 0.16,
    align_x: str = "CENTER",
    align_y: str = "CENTER",
    extrude: float = 0.001,
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
    )
    look_at(obj, camera_location, track="Z", up="Y")
    return obj


def setup_render() -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_END
    scene.frame_set(FRAME_END)
    scene.render.fps = FPS
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.film_transparent = False

    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue

    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 96
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = True
            scene.eevee.bloom_intensity = 0.035
            scene.eevee.bloom_radius = 4.5
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
            scene.eevee.gtao_distance = 5
            scene.eevee.gtao_factor = 1.2

    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - High Contrast", "High Contrast", "None"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    scene.view_settings.exposure = -0.55
    scene.view_settings.gamma = 1.0

    try:
        scene.use_nodes = True
    except Exception:
        pass
    tree = getattr(scene, "node_tree", None)
    if tree is not None:
        for node in list(tree.nodes):
            tree.nodes.remove(node)
        render_layers = tree.nodes.new(type="CompositorNodeRLayers")
        glare = tree.nodes.new(type="CompositorNodeGlare")
        glare.glare_type = "FOG_GLOW"
        glare.quality = "HIGH"
        glare.threshold = 0.72
        glare.size = 6
        composite = tree.nodes.new(type="CompositorNodeComposite")
        viewer = tree.nodes.new(type="CompositorNodeViewer")
        tree.links.new(render_layers.outputs["Image"], glare.inputs["Image"])
        tree.links.new(glare.outputs["Image"], composite.inputs["Image"])
        tree.links.new(glare.outputs["Image"], viewer.inputs["Image"])

    world = scene.world or bpy.data.worlds.new("ClaimObservatory_World")
    scene.world = world
    world.color = (0.002, 0.003, 0.006)


def setup_materials() -> dict[str, bpy.types.Material]:
    return {
        "matte_black": make_material("MAT_matte_black_observatory", (0.006, 0.008, 0.012, 1), roughness=0.82),
        "dark_wall": make_material("MAT_dark_semantic_firewall", (0.012, 0.014, 0.018, 1), roughness=0.78),
        "floor": make_material("MAT_dark_hash_ledger_floor", (0.008, 0.010, 0.013, 1), roughness=0.67),
        "glass": make_material(
            "MAT_transparent_claim_capsule_glass",
            (0.48, 0.82, 1.0, 0.22),
            alpha=0.22,
            roughness=0.02,
            transmission=0.68,
            emission=(0.08, 0.35, 0.55, 1.0),
            emission_strength=0.08,
        ),
        "cyan": make_material(
            "MAT_evidence_cyan_glow",
            (0.16, 0.74, 1.0, 1),
            emission=(0.13, 0.72, 1.0, 1),
            emission_strength=1.9,
            roughness=0.28,
        ),
        "cyan_soft": make_material(
            "MAT_soft_bluewhite_evidence",
            (0.58, 0.92, 1.0, 1),
            emission=(0.45, 0.86, 1.0, 1),
            emission_strength=0.9,
            roughness=0.34,
        ),
        "white": make_material(
            "MAT_cool_white_engraving",
            (0.82, 0.92, 1.0, 1),
            emission=(0.72, 0.88, 1.0, 1),
            emission_strength=0.78,
            roughness=0.35,
        ),
        "amber": make_material(
            "MAT_requires_review_amber",
            (1.0, 0.58, 0.18, 1),
            emission=(1.0, 0.42, 0.10, 1),
            emission_strength=1.45,
            roughness=0.36,
        ),
        "red": make_material(
            "MAT_invalidation_red_limited",
            (0.95, 0.08, 0.06, 1),
            emission=(0.95, 0.04, 0.035, 1),
            emission_strength=1.25,
            roughness=0.42,
        ),
        "weak": make_material(
            "MAT_weak_integrity_only_edge",
            (0.18, 0.29, 0.34, 1),
            emission=(0.12, 0.24, 0.31, 1),
            emission_strength=0.42,
            roughness=0.58,
        ),
        "parchment": make_material(
            "MAT_translucent_json_parchment",
            (0.70, 0.73, 0.68, 0.55),
            alpha=0.55,
            roughness=0.38,
            emission=(0.18, 0.22, 0.20, 1),
            emission_strength=0.05,
        ),
        "panel": make_material(
            "MAT_smoked_holographic_panel",
            (0.025, 0.040, 0.052, 0.42),
            alpha=0.42,
            roughness=0.5,
            transmission=0.2,
        ),
        "volume": make_volume_material("MAT_subtle_observatory_haze"),
    }


def build_claim_capsule(collections: dict[str, bpy.types.Collection], materials: dict[str, bpy.types.Material]) -> None:
    capsule = collections["ClaimCapsule"]
    evidence = collections["EvidenceArtifacts"]

    add_cylinder(capsule, "ClaimCapsule_GlassCylinder", CENTER, 1.10, 2.28, materials["glass"], vertices=160)
    add_sphere(
        capsule,
        "ClaimCapsule_GlassTopCap",
        (CENTER.x, CENTER.y, CENTER.z + 1.14),
        1.10,
        materials["glass"],
        scale=(1.0, 1.0, 0.52),
        segments=64,
        ring_count=24,
    )
    add_sphere(
        capsule,
        "ClaimCapsule_GlassBottomCap",
        (CENTER.x, CENTER.y, CENTER.z - 1.14),
        1.10,
        materials["glass"],
        scale=(1.0, 1.0, 0.52),
        segments=64,
        ring_count=24,
    )

    for idx, z in enumerate((CENTER.z - 1.12, CENTER.z - 0.52, CENTER.z, CENTER.z + 0.52, CENTER.z + 1.12), 1):
        add_torus(capsule, f"ClaimCapsule_CyanSealRing_{idx:02d}", (0, 0, z), 1.105, 0.006, materials["cyan_soft"])

    for idx, angle in enumerate((0, 60, 120, 180, 240, 300), 1):
        theta = math.radians(angle)
        x = math.cos(theta) * 1.11
        y = math.sin(theta) * 1.11
        add_curve(
            capsule,
            f"ClaimCapsule_VerticalContainmentLine_{idx:02d}",
            [Vector((x, y, CENTER.z - 1.08)), Vector((x, y, CENTER.z + 1.08))],
            materials["cyan_soft"],
            bevel_depth=0.0035,
        )

    json_sheets = [
        (
            "ClaimCapsule_JSONPlane_ClaimEnvelope",
            "cc.claim_envelope.v1\nclaim_id: cc-demo-claim\nverifier_schema: active\nsupport_edges: 5",
            CENTER.z + 0.47,
            -5,
        ),
        (
            "ClaimCapsule_JSONPlane_GovernanceAudit",
            "cc/claim-governance-audit.v1\nverdict: PASS\ncaveat: verifier rules only\nreview_edges: 1",
            CENTER.z + 0.14,
            4,
        ),
        (
            "ClaimCapsule_JSONPlane_Report",
            "cc.report.v0.3.1\nclaim_level: bounded_empirical\nnon_claims: attached\nreceipt: sha256",
            CENTER.z - 0.19,
            -2,
        ),
        (
            "ClaimCapsule_JSONPlane_Manifest",
            "capsule_manifest.json\nfixed_now: replayed\nseed: deterministic\nMeaning survives replay",
            CENTER.z - 0.52,
            7,
        ),
    ]
    for idx, (name, text, z, angle) in enumerate(json_sheets, 1):
        rot = (0.0, 0.0, math.radians(angle))
        sheet = add_cube(
            evidence,
            name,
            (0.0, -0.075 - idx * 0.006, z),
            (1.43, 0.014, 0.42),
            materials["parchment"],
            rotation=rot,
            bevel=0.012,
            bevel_segments=4,
        )
        sheet["claim_observatory_role"] = "layered JSON-like evidence sheet"
        text_obj = add_text(
            evidence,
            f"{name}_Microtext",
            text,
            (-0.62, -0.105 - idx * 0.006, z + 0.055),
            materials["white"],
            size=0.048,
            rotation=(math.radians(90), 0.0, math.radians(angle)),
            align_x="LEFT",
            align_y="CENTER",
            extrude=0.0007,
            line_spacing=0.84,
        )
        text_obj["claim_observatory_role"] = "visible schema tag and verifier field"

    random.seed(42)
    atoms: list[Vector] = []
    for xi in range(4):
        for yi in range(3):
            for zi in range(4):
                loc = Vector(
                    (
                        -0.48 + xi * 0.32 + random.uniform(-0.025, 0.025),
                        0.24 + yi * 0.17 + random.uniform(-0.018, 0.018),
                        CENTER.z - 0.50 + zi * 0.30 + random.uniform(-0.025, 0.025),
                    )
                )
                atoms.append(loc)
                add_sphere(
                    evidence,
                    f"EvidenceArtifacts_FiniteAtomFrechetPoint_{xi}_{yi}_{zi}",
                    loc,
                    0.020,
                    materials["cyan_soft"],
                    segments=12,
                    ring_count=6,
                )
    for idx, atom in enumerate(atoms[::3]):
        neighbor = atoms[(idx * 3 + 7) % len(atoms)]
        add_curve(
            evidence,
            f"EvidenceArtifacts_FiniteAtomFrechetWeakLink_{idx:02d}",
            [atom, neighbor],
            materials["weak"],
            bevel_depth=0.002,
        )

    add_facing_text(
        capsule,
        "ClaimCapsule_Label_EvidenceBoundClaim",
        "Evidence-bound claim",
        (-0.46, -1.28, CENTER.z + 1.18),
        materials["white"],
        Vector((5.8, -8.0, 4.4)),
        size=0.072,
    )
    add_facing_text(
        capsule,
        "ClaimCapsule_Label_PassUnderVerifierRules",
        "PASS under verifier rules",
        (-0.58, -1.20, CENTER.z - 1.32),
        materials["cyan_soft"],
        Vector((5.8, -8.0, 4.4)),
        size=0.095,
        align_x="LEFT",
    )
    add_facing_text(
        capsule,
        "ClaimCapsule_Label_NotDeploymentSafetyProof",
        "Not a deployment-safety proof",
        (-0.58, -1.20, CENTER.z - 1.50),
        materials["amber"],
        Vector((5.8, -8.0, 4.4)),
        size=0.073,
        align_x="LEFT",
    )
    add_facing_text(
        capsule,
        "ClaimCapsule_Label_ReceiptIntegrityNotSafety",
        "Receipt integrity ≠ safety",
        (0.36, -1.18, CENTER.z - 1.50),
        materials["white"],
        Vector((5.8, -8.0, 4.4)),
        size=0.066,
        align_x="LEFT",
    )


def build_support_graph(collections: dict[str, bpy.types.Collection], materials: dict[str, bpy.types.Material]) -> None:
    graph = collections["SupportGraph"]
    camera_hint = Vector((5.8, -8.0, 4.4))
    nodes = {
        "claim_fragment": (Vector((-0.55, -0.25, CENTER.z + 0.45)), materials["white"], "claim fragment"),
        "bounds": (Vector((-2.05, -0.72, CENTER.z + 0.74)), materials["cyan"], "bounds.json"),
        "upper": (Vector((1.92, -0.56, CENTER.z + 0.93)), materials["cyan"], "extremal_upper"),
        "lower": (Vector((-1.62, 0.86, CENTER.z - 0.20)), materials["cyan_soft"], "extremal_lower"),
        "manifest": (Vector((1.70, 0.86, CENTER.z - 0.25)), materials["weak"], "manifest receipt"),
        "confirmatory": (Vector((0.28, -2.12, CENTER.z + 0.10)), materials["cyan"], "confirmatory tests"),
        "review": (Vector((2.35, -0.05, CENTER.z - 0.85)), materials["amber"], "human review"),
    }
    for key, (loc, material, label) in nodes.items():
        radius = 0.078 if key != "claim_fragment" else 0.095
        add_sphere(graph, f"SupportGraph_Node_{key}", loc, radius, material, segments=24, ring_count=12)
        add_facing_text(
            graph,
            f"SupportGraph_NodeLabel_{key}",
            label,
            (loc.x, loc.y, loc.z + 0.18),
            material if key != "manifest" else materials["weak"],
            camera_hint,
            size=0.055,
        )

    edge_specs = [
        ("bounds", "claim_fragment", "bounds", materials["cyan_soft"], 0.011, False),
        ("lower", "claim_fragment", "qualifies", materials["cyan_soft"], 0.008, False),
        ("manifest", "claim_fragment", "integrity_binds", materials["weak"], 0.006, True),
        ("confirmatory", "claim_fragment", "confirmatory_tests", materials["cyan"], 0.013, False),
        ("review", "claim_fragment", "requires_review", materials["amber"], 0.011, False),
        ("upper", "claim_fragment", "bounds", materials["cyan_soft"], 0.011, False),
    ]
    for start_key, end_key, label, material, width, dashed in edge_specs:
        start = nodes[start_key][0]
        end = nodes[end_key][0]
        midpoint = start.lerp(end, 0.5)
        lift = Vector((0.0, 0.0, 0.16 + 0.05 * math.sin(len(label))))
        if dashed:
            add_dashed_curve(graph, f"SupportGraph_Edge_{label}", start, end, material, bevel_depth=width)
        else:
            add_curve(graph, f"SupportGraph_Edge_{label}", [start, midpoint + lift, end], material, bevel_depth=width)
        add_facing_text(
            graph,
            f"SupportGraph_EdgeLabel_{label}",
            label,
            (midpoint.x, midpoint.y, midpoint.z + 0.23),
            material,
            camera_hint,
            size=0.052,
        )

    add_torus(
        graph,
        "SupportGraph_OrbitReferenceRing",
        (0.0, 0.0, CENTER.z + 0.05),
        2.24,
        0.004,
        materials["weak"],
        rotation=(math.radians(82), 0.0, math.radians(18)),
        major_segments=192,
    )


def build_non_claims_wall(collections: dict[str, bpy.types.Collection], materials: dict[str, bpy.types.Material]) -> None:
    wall = collections["NonClaimsWall"]
    add_cube(
        wall,
        "NonClaimsWall_SemanticFirewallMonolith",
        (0.0, 2.45, 2.55),
        (5.9, 0.16, 3.10),
        materials["dark_wall"],
        bevel=0.025,
        bevel_segments=5,
    )
    for idx, x in enumerate((-2.60, -1.85, -1.10, -0.35, 0.40, 1.15, 1.90, 2.65), 1):
        add_curve(
            wall,
            f"NonClaimsWall_VerticalGuardLine_{idx:02d}",
            [Vector((x, 2.345, 1.05)), Vector((x, 2.345, 4.03))],
            materials["weak"],
            bevel_depth=0.0025,
        )

    add_text(
        wall,
        "NonClaimsWall_Title",
        "Non-Claims Wall",
        (-2.62, 2.335, 3.86),
        materials["amber"],
        size=0.165,
        rotation=(math.radians(90), 0.0, 0.0),
        align_x="LEFT",
        extrude=0.0015,
    )
    add_text(
        wall,
        "NonClaimsWall_EngravedFirewallText",
        "This claim does NOT say:\n"
        "deployment safety\n"
        "external validity\n"
        "release approval",
        (-2.62, 2.330, 3.25),
        materials["white"],
        size=0.135,
        rotation=(math.radians(90), 0.0, 0.0),
        align_x="LEFT",
        align_y="CENTER",
        extrude=0.001,
        line_spacing=0.92,
    )
    add_text(
        wall,
        "NonClaimsWall_AttachedLabel",
        "Non-claims attached",
        (1.08, 2.330, 3.82),
        materials["cyan_soft"],
        size=0.125,
        rotation=(math.radians(90), 0.0, 0.0),
        align_x="LEFT",
        extrude=0.001,
    )
    add_text(
        wall,
        "NonClaimsWall_CaveatLabel",
        "Semantic firewall: limits remain visible",
        (1.08, 2.330, 3.52),
        materials["weak"],
        size=0.075,
        rotation=(math.radians(90), 0.0, 0.0),
        align_x="LEFT",
        extrude=0.001,
    )


def build_decay_clock(collections: dict[str, bpy.types.Collection], materials: dict[str, bpy.types.Material]) -> None:
    decay = collections["DecayClock"]
    y = -0.18
    add_curve(decay, "DecayClock_FreshSegment", arc_points(CENTER, 2.02, 36, 162, plane="XZ", y=y), materials["cyan"], bevel_depth=0.018)
    add_curve(decay, "DecayClock_DegradedSegment", arc_points(CENTER, 2.02, 162, 292, plane="XZ", y=y), materials["amber"], bevel_depth=0.015)
    add_curve(decay, "DecayClock_ExpiredSegment", arc_points(CENTER, 2.02, 292, 396, plane="XZ", y=y), materials["red"], bevel_depth=0.012)
    add_curve(decay, "DecayClock_OuterQuietRing", arc_points(CENTER, 2.12, 0, 360, plane="XZ", y=y + 0.025, steps=180), materials["weak"], bevel_depth=0.004)

    marker_angle = math.radians(72)
    marker = Vector((CENTER.x + math.cos(marker_angle) * 2.02, y - 0.02, CENTER.z + math.sin(marker_angle) * 2.02))
    add_sphere(decay, "DecayClock_CurrentFreshMarker", marker, 0.075, materials["white"], segments=24, ring_count=12)
    add_curve(
        decay,
        "DecayClock_CurrentFreshMarkerNeedle",
        [CENTER + Vector((0, y, 0)), marker],
        materials["white"],
        bevel_depth=0.004,
    )
    camera_hint = Vector((5.8, -8.0, 4.4))
    add_facing_text(decay, "DecayClock_Label_Fresh", "Fresh", (1.35, -0.58, 4.50), materials["cyan"], camera_hint, size=0.09)
    add_facing_text(decay, "DecayClock_Label_Degraded", "Degraded", (-2.18, -0.55, 2.82), materials["amber"], camera_hint, size=0.083)
    add_facing_text(decay, "DecayClock_Label_Expired", "Expired", (1.28, -0.55, 1.18), materials["red"], camera_hint, size=0.078)
    add_facing_text(
        decay,
        "DecayClock_Label_FreshDegradedExpired",
        "Fresh / Degraded / Expired",
        (0.02, -0.63, 0.95),
        materials["white"],
        camera_hint,
        size=0.075,
    )


def build_replay_manifest(collections: dict[str, bpy.types.Collection], materials: dict[str, bpy.types.Material]) -> None:
    floor = collections["ReplayManifest"]
    add_cylinder(floor, "ReplayManifest_CircularHashLedgerFloor", (0.0, 0.0, -0.035), 5.85, 0.07, materials["floor"], vertices=192)
    for idx, radius in enumerate((1.38, 2.52, 3.70, 5.05), 1):
        add_torus(
            floor,
            f"ReplayManifest_EngravedLedgerRing_{idx:02d}",
            (0.0, 0.0, 0.022),
            radius,
            0.0038,
            materials["weak"] if idx != 2 else materials["cyan_soft"],
            major_segments=192,
            minor_segments=6,
        )

    rows = [
        ("claim_envelope.json", "18,930 B", "sha256:7f20f937"),
        ("claim_governance_audit.json", "11,279 B", "sha256:62ab6600"),
        ("capsule_manifest.json", "3,016 B", "sha256:d6a29afe"),
        ("bounds.json", "1,224 B", "sha256:ce2cbba1"),
        ("extremal_lower.json", "4,347 B", "sha256:71830fc9"),
        ("extremal_upper.json", "4,345 B", "sha256:42cd2764"),
    ]
    for idx, (filename, byte_count, digest) in enumerate(rows):
        angle = math.radians(25 + idx * 54)
        radius = 3.65 if idx % 2 else 4.32
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius
        body = f"{filename}  {byte_count}\n{digest}"
        add_text(
            floor,
            f"ReplayManifest_FloorEngraving_{idx+1:02d}_{filename.replace('.', '_')}",
            body,
            (x, y, 0.045),
            materials["weak"] if idx % 2 else materials["white"],
            size=0.105,
            rotation=(0.0, 0.0, angle + math.pi / 2),
            align_x="CENTER",
            align_y="CENTER",
            extrude=0.0008,
            line_spacing=0.82,
        )

    add_text(
        floor,
        "ReplayManifest_Title_ReplayableCapsule",
        "Replayable capsule",
        (-1.82, -3.08, 0.065),
        materials["cyan_soft"],
        size=0.16,
        rotation=(0.0, 0.0, math.radians(8)),
        align_x="LEFT",
        extrude=0.001,
    )
    add_text(
        floor,
        "ReplayManifest_Label_MeaningSurvivesReplay",
        "Meaning survives replay",
        (-1.82, -3.42, 0.065),
        materials["white"],
        size=0.105,
        rotation=(0.0, 0.0, math.radians(8)),
        align_x="LEFT",
        extrude=0.001,
    )


def build_human_review(collections: dict[str, bpy.types.Collection], materials: dict[str, bpy.types.Material]) -> None:
    human = collections["HumanReview"]
    add_cube(human, "HumanReview_DistantReviewerSeat_Base", (3.95, 2.85, 0.42), (0.78, 0.70, 0.14), materials["dark_wall"], bevel=0.025)
    add_cube(human, "HumanReview_DistantReviewerSeat_Back", (3.95, 3.12, 0.96), (0.80, 0.12, 0.92), materials["dark_wall"], bevel=0.025)
    add_cube(human, "HumanReview_AccountabilityConsole", (3.10, 2.30, 0.74), (1.18, 0.42, 0.36), materials["matte_black"], rotation=(0.0, 0.0, math.radians(-10)), bevel=0.02)
    add_cube(human, "HumanReview_ConsoleAmberReviewLight", (3.05, 2.08, 0.96), (0.72, 0.020, 0.035), materials["amber"], rotation=(0.0, 0.0, math.radians(-10)), bevel=0.006)
    add_facing_text(
        human,
        "HumanReview_Label",
        "Human Review",
        (3.12, 1.96, 1.25),
        materials["amber"],
        Vector((5.8, -8.0, 4.4)),
        size=0.115,
    )
    add_facing_text(
        human,
        "HumanReview_AccountabilityNote",
        "accountable reviewer\nnot a rubber stamp",
        (3.12, 1.96, 1.02),
        materials["weak"],
        Vector((5.8, -8.0, 4.4)),
        size=0.061,
    )


def build_orbiting_panels(collections: dict[str, bpy.types.Collection], materials: dict[str, bpy.types.Material]) -> None:
    target_collections = {
        "Claim Envelope": collections["ClaimCapsule"],
        "Governance Audit": collections["ClaimCapsule"],
        "Evidence Body": collections["EvidenceArtifacts"],
        "Support Graph": collections["SupportGraph"],
        "Non-Claims Wall": collections["NonClaimsWall"],
        "Decay Clock": collections["DecayClock"],
        "Replay Manifest": collections["ReplayManifest"],
        "Human Review": collections["HumanReview"],
    }
    labels = list(target_collections.keys())
    radius = 4.34
    for idx, label in enumerate(labels):
        angle = math.radians(20 + idx * (360 / len(labels)))
        z = CENTER.z + 0.30 + 0.42 * math.sin(angle * 1.7)
        loc = Vector((math.cos(angle) * radius, math.sin(angle) * radius, z))
        yaw = angle - math.pi / 2
        collection = target_collections[label]
        compact = label.replace("-", "").replace(" ", "")
        add_cube(
            collection,
            f"OrbitPanel_{compact}_SmokedGlass",
            loc,
            (1.55, 0.040, 0.40),
            materials["panel"],
            rotation=(0.0, 0.0, yaw),
            bevel=0.018,
            bevel_segments=4,
        )
        direction_to_center = Vector((-math.cos(angle), -math.sin(angle), 0.0))
        text_loc = loc + direction_to_center * 0.045 + Vector((0.0, 0.0, 0.015))
        add_facing_text(
            collection,
            f"OrbitPanel_{compact}_Label",
            label,
            text_loc,
            materials["white"] if label not in {"Human Review", "Decay Clock"} else materials["amber"],
            Vector((5.8, -8.0, 4.4)),
            size=0.105 if len(label) < 14 else 0.082,
            align_x="CENTER",
            extrude=0.001,
        )
        add_curve(
            collection,
            f"OrbitPanel_{compact}_EvidenceTether",
            [loc + direction_to_center * 0.82, CENTER + Vector((0.0, 0.0, 0.18 * math.sin(angle)))],
            materials["weak"],
            bevel_depth=0.0028,
        )


def build_hero_text(collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]) -> None:
    camera_hint = Vector((5.8, -8.0, 4.4))
    add_facing_text(
        collection,
        "HeroText_NotSafetyScores",
        "Not safety scores.",
        (-0.52, -2.56, 4.46),
        materials["white"],
        camera_hint,
        size=0.108,
        align_x="CENTER",
    )
    add_facing_text(
        collection,
        "HeroText_EvidenceBoundClaims",
        "Evidence-bound claims.",
        (-0.46, -2.56, 4.24),
        materials["cyan_soft"],
        camera_hint,
        size=0.126,
        align_x="CENTER",
    )
    add_facing_text(
        collection,
        "HeroText_AClaimIsNotASentence",
        "A claim is a bounded artifact with receipts, non-claims, decay, challenge surfaces, and replay.",
        (-0.38, -2.58, 4.02),
        materials["weak"],
        camera_hint,
        size=0.038,
    )


def build_lighting(collections: dict[str, bpy.types.Collection], materials: dict[str, bpy.types.Material]) -> None:
    lighting = collections["Lighting"]
    add_cube(lighting, "Lighting_SubtleVolumetricHazeBox", (0.0, 0.0, 2.6), (12.0, 12.0, 7.0), materials["volume"])

    light_specs = [
        ("Lighting_KeySoftbox_CapsuleTop", "AREA", (0.0, -3.2, 6.9), 260, (3.4, 2.2)),
        ("Lighting_CyanEvidenceRim_Left", "POINT", (-3.8, -2.8, 3.9), 180, None),
        ("Lighting_CyanEvidenceRim_Right", "POINT", (3.4, -2.2, 4.0), 130, None),
        ("Lighting_AmberReviewPractical", "POINT", (3.05, 1.95, 1.42), 95, None),
        ("Lighting_LowLedgerGrazingLight", "AREA", (0.0, -4.8, 0.85), 95, (5.4, 1.1)),
    ]
    for name, light_type, loc, energy, size in light_specs:
        data = bpy.data.lights.new(name, type=light_type)
        data.energy = energy
        if light_type == "AREA" and size:
            data.size = size[0]
            if hasattr(data, "size_y"):
                data.size_y = size[1]
        if name.endswith("ReviewPractical"):
            data.color = (1.0, 0.58, 0.22)
        elif "Cyan" in name:
            data.color = (0.45, 0.80, 1.0)
        else:
            data.color = (0.76, 0.88, 1.0)
        obj = bpy.data.objects.new(name, data)
        obj.location = loc
        lighting.objects.link(obj)
        if light_type == "AREA":
            look_at(obj, CENTER)


def build_camera_rig(collections: dict[str, bpy.types.Collection]) -> None:
    rig = collections["CameraRig"]
    target = bpy.data.objects.new("CameraRig_LookAt_ClaimCapsule", None)
    target.empty_display_type = "PLAIN_AXES"
    target.empty_display_size = 0.35
    target.location = CENTER
    rig.objects.link(target)

    camera_data = bpy.data.cameras.new("CameraRig_MainCinematicCamera_Data")
    camera = bpy.data.objects.new("CameraRig_MainCinematicCamera", camera_data)
    rig.objects.link(camera)
    camera_data.lens = 48
    camera_data.dof.use_dof = True
    camera_data.dof.focus_object = target
    camera_data.dof.aperture_fstop = 5.6
    bpy.context.scene.camera = camera

    shots = [
        (1, "Shot 1 - Wide reveal of dark observatory", Vector((7.4, -9.3, 5.25)), CENTER + Vector((0, 0, 0.05)), 34),
        (72, "Shot 2 - Push toward glowing Claim Capsule", Vector((5.35, -6.75, 4.45)), CENTER + Vector((0, 0, 0.12)), 46),
        (144, "Shot 3 - Orbit support graph edges", Vector((-4.65, -6.15, 3.60)), CENTER + Vector((0.20, -0.20, 0.05)), 58),
        (216, "Shot 4 - Pan to Non-Claims Wall", Vector((3.10, -6.35, 3.34)), Vector((-0.65, 2.28, 3.05)), 54),
        (288, "Shot 5 - Close-up on Decay Clock", Vector((2.50, -4.90, 4.55)), Vector((0.92, -0.18, 4.38)), 70),
        (360, "Shot 6 - Final hero frame", Vector((5.85, -7.75, 4.35)), CENTER + Vector((0, 0, 0.12)), 50),
    ]
    scene = bpy.context.scene
    for frame, marker_name, loc, target_loc, lens in shots:
        scene.frame_set(frame)
        camera.location = loc
        look_at(camera, target_loc)
        camera.data.lens = lens
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
        camera.data.keyframe_insert("lens", frame=frame)
        marker = scene.timeline_markers.new(marker_name, frame=frame)
        marker.camera = camera

    scene.frame_set(FRAME_END)


def build_scene() -> None:
    reset_scene()
    setup_render()
    materials = setup_materials()
    collections = {
        name: new_collection(name)
        for name in (
            "ClaimCapsule",
            "EvidenceArtifacts",
            "SupportGraph",
            "NonClaimsWall",
            "DecayClock",
            "ReplayManifest",
            "HumanReview",
            "CameraRig",
            "Lighting",
        )
    }

    build_replay_manifest(collections, materials)
    build_claim_capsule(collections, materials)
    build_support_graph(collections, materials)
    build_non_claims_wall(collections, materials)
    build_decay_clock(collections, materials)
    build_human_review(collections, materials)
    build_orbiting_panels(collections, materials)
    build_hero_text(collections["ClaimCapsule"], materials)
    build_lighting(collections, materials)
    build_camera_rig(collections)

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-still", action="store_true", help="Render the final hero frame after creating the .blend.")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    build_scene()
    if args.render_still:
        scene = bpy.context.scene
        scene.frame_set(FRAME_END)
        scene.render.filepath = str(HERO_RENDER_PATH)
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
