bl_info = {
    "name": "KUKA KPL/KRL -> Empty Animation (TCP Stand-in)",
    "author": "ChatGPT",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),  # should also work on Blender 5.x unless API changes drastically
    "location": "View3D > Sidebar > Animation > KUKA Import",
    "description": "Parse simple KUKA pose data and keyframe a single Empty (TCP stand-in).",
    "category": "Animation",
}

import bpy
import re
import math
from bpy.props import (
    StringProperty,
    PointerProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    BoolProperty,
)


# --- Parsing helpers ---

# Captures blocks like: {X 10, Y 20, Z 30, A 0, B 0, C 0}
POSE_BLOCK_RE = re.compile(r"\{[^}]*\}", re.IGNORECASE)

# Captures key-value pairs inside a pose block:
# X 10.0, Y -20, Z 30, A 0, B 90, C 180
KV_RE = re.compile(
    r"(?P<key>[XYZABC])\s*[:=]?\s*(?P<val>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Captures named pose assignments: P1={...}
NAMED_POSE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<block>\{[^}]*\})",
    re.IGNORECASE,
)

# Captures motion commands referencing a named pose: LIN P1 / PTP P1
MOTION_REF_RE = re.compile(
    r"^\s*(LIN|PTP)\s+(?P<name>[A-Za-z_]\w*)\b",
    re.IGNORECASE,
)

# Captures CIRC commands with two pose blocks: CIRC {...}, {...}
CIRC_RE = re.compile(
    r"^\s*CIRC\s+(?P<aux>\{[^}]*\})\s*,\s*(?P<end>\{[^}]*\})",
    re.IGNORECASE,
)

# Captures WAIT SEC commands: WAIT SEC 1.0
WAIT_SEC_RE = re.compile(
    r"^\s*WAIT\s+SEC\s+(?P<seconds>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_pose_block(block: str):
    """Return dict with X,Y,Z,A,B,C if present in the block."""
    pose = {}
    for m in KV_RE.finditer(block):
        key = m.group("key").upper()
        val = float(m.group("val"))
        pose[key] = val
    # require at least XYZ; rotations optional but expected
    if not all(k in pose for k in ("X", "Y", "Z")):
        return None
    # fill missing rotations with 0
    for k in ("A", "B", "C"):
        pose.setdefault(k, 0.0)
    return pose


def parse_kuka_file(text: str):
    """
    Parse:
    - named poses like P1={X...,Y...,Z...,A...,B...,C...}
    - motion lines containing inline { ... }
    - motion lines referencing named pose (LIN P1 / PTP P1)
    - CIRC commands with two pose blocks (auxiliary and end points)
    - WAIT SEC commands for pauses
    Returns list of commands (pose dicts or pause commands) in encountered order.
    Each pause is represented as {"type": "pause", "seconds": float}.
    """
    named = {}
    commands = []

    lines = text.splitlines()
    for line in lines:
        # strip comments (KUKA uses ';' often)
        line_wo_comment = line.split(";")[0].strip()
        if not line_wo_comment:
            continue

        # 1) Named pose assignment
        m_named = NAMED_POSE_RE.match(line_wo_comment)
        if m_named:
            nm = m_named.group("name")
            blk = m_named.group("block")
            pose = parse_pose_block(blk)
            if pose:
                named[nm.upper()] = pose
            continue

        # 2) WAIT SEC command
        m_wait = WAIT_SEC_RE.match(line_wo_comment)
        if m_wait:
            seconds = float(m_wait.group("seconds"))
            commands.append({"type": "pause", "seconds": seconds})
            continue

        # 3) CIRC command with two pose blocks (auxiliary point, end point)
        m_circ = CIRC_RE.match(line_wo_comment)
        if m_circ:
            aux_block = m_circ.group("aux")
            end_block = m_circ.group("end")
            aux_pose = parse_pose_block(aux_block)
            end_pose = parse_pose_block(end_block)
            if aux_pose and end_pose:
                # Add both auxiliary and end points as keyframes
                commands.append(aux_pose)
                commands.append(end_pose)
            continue

        # 4) Motion line with inline pose block
        if re.search(r"\b(LIN|PTP)\b", line_wo_comment, re.IGNORECASE):
            blk_m = POSE_BLOCK_RE.search(line_wo_comment)
            if blk_m:
                pose = parse_pose_block(blk_m.group(0))
                if pose:
                    commands.append(pose)
                    continue

            # 5) Motion line referencing named pose
            m_ref = MOTION_REF_RE.match(line_wo_comment)
            if m_ref:
                nm = m_ref.group("name").upper()
                if nm in named:
                    commands.append(named[nm])
                continue

    return commands


# --- Blender side ---

def ensure_empty(obj):
    return obj and obj.type == "EMPTY"


def set_empty_pose(obj, pose, mm_to_m: float, rot_order: str):
    # KUKA XYZ in mm (typically), convert to meters
    x = pose["X"] * mm_to_m
    y = pose["Y"] * mm_to_m
    z = pose["Z"] * mm_to_m

    # KUKA ABC in degrees
    a = math.radians(pose.get("A", 0.0))
    b = math.radians(pose.get("B", 0.0))
    c = math.radians(pose.get("C", 0.0))

    obj.location = (x, y, z)
    obj.rotation_mode = rot_order
    # NOTE: This assumes A,B,C map directly onto Euler components in your chosen order.
    # Many KUKA setups use an ABC convention that may need remapping.
    obj.rotation_euler = (a, b, c)


def keyframe_pose(obj, frame: int):
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)


class KUKAImporterSettings(bpy.types.PropertyGroup):
    filepath: StringProperty(
        name="KUKA File",
        subtype="FILE_PATH",
        description="Path to KUKA KPL/KRL text file",
    ) # type: ignore

    target: PointerProperty(
        name="Target Empty",
        type=bpy.types.Object,
        description="Empty that will receive keyframes (TCP stand-in)",
    ) # type: ignore

    start_frame: IntProperty(
        name="Start Frame",
        default=1,
        min=0,
    ) # type: ignore

    frame_step: IntProperty(
        name="Frame Step",
        default=10,
        min=1,
        description="Frames to advance per pose (simple timing)",
    ) # type: ignore

    mm_scale: FloatProperty(
        name="MM to M Scale",
        default=0.001,
        min=0.000001,
        description="Conversion factor from mm to meters (0.001 is typical)",
    ) # type: ignore

    global_scale: FloatProperty(
        name="Global Scale",
        default=1.0,
        min=0.000001,
        description="Extra scaling applied after unit conversion",
    )   # type: ignore

    rot_order: EnumProperty(
        name="Euler Order",
        items=[
            ("XYZ", "XYZ", ""),
            ("XZY", "XZY", ""),
            ("YXZ", "YXZ", ""),
            ("YZX", "YZX", ""),
            ("ZXY", "ZXY", ""),
            ("ZYX", "ZYX", ""),
        ],
        default="ZYX",
        description="Euler rotation order for the Empty",
    ) # type: ignore

    clear_existing: BoolProperty(
        name="Clear Existing Animation",
        default=True,
        description="Remove existing animation data on the target before importing",
    ) # type: ignore


class ANIM_OT_import_kuka_kpl(bpy.types.Operator):
    bl_idname = "anim.import_kuka_kpl_to_empty"
    bl_label = "Import KUKA -> Empty Animation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.kuka_importer_settings

        if not s.filepath:
            self.report({"ERROR"}, "No file selected.")
            return {"CANCELLED"}

        if not ensure_empty(s.target):
            self.report({"ERROR"}, "Target must be an Empty.")
            return {"CANCELLED"}

        try:
            with open(bpy.path.abspath(s.filepath), "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read file: {e}")
            return {"CANCELLED"}

        commands = parse_kuka_file(text)
        if not commands:
            self.report({"ERROR"}, "No motion commands found. Check file format.")
            return {"CANCELLED"}

        obj = s.target

        if s.clear_existing:
            obj.animation_data_clear()

        mm_to_m = s.mm_scale * s.global_scale

        frame = s.start_frame
        pose_count = 0
        pause_count = 0
        last_pose = None
        
        for cmd in commands:
            if isinstance(cmd, dict) and cmd.get("type") == "pause":
                # WAIT SEC command - hold current position for specified duration
                pause_count += 1
                seconds = cmd["seconds"]
                # Convert seconds to frames (assuming 24 fps by default)
                # You could make fps a configurable property if needed
                fps = context.scene.render.fps
                pause_frames = int(seconds * fps)
                
                if last_pose and pause_frames > 0:
                    # Create a keyframe at the end of the pause with the same pose
                    frame += pause_frames
                    set_empty_pose(obj, last_pose, mm_to_m=mm_to_m, rot_order=s.rot_order)
                    keyframe_pose(obj, frame)
            else:
                # Regular pose command
                pose_count += 1
                set_empty_pose(obj, cmd, mm_to_m=mm_to_m, rot_order=s.rot_order)
                keyframe_pose(obj, frame)
                last_pose = cmd
                frame += s.frame_step

        msg = f"Imported {pose_count} poses"
        if pause_count > 0:
            msg += f" and {pause_count} pauses"
        msg += f" to {obj.name}."
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class VIEW3D_PT_kuka_import_panel(bpy.types.Panel):
    bl_label = "KUKA Import"
    bl_idname = "VIEW3D_PT_kuka_import_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Animation"

    def draw(self, context):
        layout = self.layout
        s = context.scene.kuka_importer_settings

        layout.prop(s, "filepath")
        layout.prop(s, "target")

        col = layout.column(align=True)
        col.prop(s, "start_frame")
        col.prop(s, "frame_step")

        col = layout.column(align=True)
        col.prop(s, "mm_scale")
        col.prop(s, "global_scale")

        layout.prop(s, "rot_order")
        layout.prop(s, "clear_existing")

        layout.operator(ANIM_OT_import_kuka_kpl.bl_idname, icon="IMPORT")


classes = (
    KUKAImporterSettings,
    ANIM_OT_import_kuka_kpl,
    VIEW3D_PT_kuka_import_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.kuka_importer_settings = PointerProperty(type=KUKAImporterSettings)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.kuka_importer_settings


if __name__ == "__main__":
    register()
