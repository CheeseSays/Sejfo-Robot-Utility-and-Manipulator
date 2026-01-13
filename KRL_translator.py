import bpy
import re
import math
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup, Operator
from bpy.path import abspath
from . CONSTANTS import POSE_BLOCK_RE, KV_RE, NAMED_POSE_RE, MOTION_REF_RE, CIRC_RE, WAIT_SEC_RE

def parse_pose_block(block: str) -> dict:
    """Parses a pose block string and returns a dictionary of joint values."""
    pose = {}
    for match in KV_RE.finditer(block):
        key = match.group("key").upper()
        value = float(match.group("val"))
        pose[key] = value
    
    if not all(k in pose for k in ("X", "Y", "Z")):
        return None  # Incomplete pose
    for k in ("A", "B", "C"):
        pose.setdefault(k, 0.0)
    return pose

def parse_program(text: str) -> list:
    """Parses a KRL program text and returns a list of motion commands."""

    named = {}
    commands = []

    lines = text.splitlines()
    for line in lines:
        line_wo_comments = line.split(';')[0].strip()
        if not line_wo_comments:
            continue

        # Named pose assignment
        named_match = NAMED_POSE_RE.match(line_wo_comments)
        if named_match:
            name = named_match.group("name")
            block = named_match.group("block")
            pose = parse_pose_block(block)
            if pose:
                named[name.upper()] = pose
            continue
        
        # Wait command
        wait_match = WAIT_SEC_RE.match(line_wo_comments)
        if wait_match:
            seconds = float(wait_match.group("SEC"))
            commands.append({"type": "WAIT", "SEC": seconds})
            continue

        # Circular motion
        circ_match = CIRC_RE.match(line_wo_comments)
        if circ_match:
            aux_block = circ_match.group("aux")
            end_block = circ_match.group("end")
            aux_pose = parse_pose_block(aux_block)
            end_pose = parse_pose_block(end_block)
            if aux_pose and end_pose:
                commands.append(aux_pose)
                commands.append(end_pose)
            continue

        # Motion line
        if re.search(r"\b(LIN|PTP)\b", line_wo_comments, re.IGNORECASE):
            match_block = POSE_BLOCK_RE.search(line_wo_comments)
            if match_block:
                pose = parse_pose_block(match_block.group(0))
                if pose:
                    commands.append(pose)
                    continue
            match_ref = MOTION_REF_RE.match(line_wo_comments)
            if match_ref:
                name = match_ref.group("name").upper()
                if name in named:
                    commands.append(named[name])
                continue
    return commands

def ensure_empty(obj):
    return obj and obj.type == 'EMPTY'

def set_empty_pose(obj, pose, mm_to_m: float, rot_order: str, robot_base=None):
    """Set pose on the target empty, optionally relative to robot_base."""
    import mathutils
    
    x = pose['X'] * mm_to_m
    y = pose['Y'] * mm_to_m
    z = pose['Z'] * mm_to_m

    a = math.radians(pose.get('A', 0))
    b = math.radians(pose.get('B', 0))
    c = math.radians(pose.get('C', 0))

    # Create local transformation
    local_location = mathutils.Vector((x, y, z))
    local_rotation = mathutils.Euler((a, b, c), rot_order)
    
    if robot_base:
        # Transform from robot base local space to world space
        base_matrix = robot_base.matrix_world
        
        # Create transformation matrix from local pose
        local_matrix = mathutils.Matrix.LocRotScale(
            local_location,
            local_rotation,
            None
        )
        
        # Apply robot base transformation to get world position
        world_matrix = base_matrix @ local_matrix
        
        obj.location = world_matrix.to_translation()
        obj.rotation_mode = rot_order
        obj.rotation_euler = world_matrix.to_euler(rot_order)
    else:
        # No robot base, use world coordinates directly
        obj.location = local_location
        obj.rotation_mode = rot_order
        obj.rotation_euler = local_rotation

def keyframe_pose(obj, frame: int):
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)

class KRLImporterSettings(PropertyGroup):
    filepath: StringProperty(
        name="File Path",
        description="Path to the KRL program file. Currently only supports XYZABC statements.",
        default="",
        subtype='FILE_PATH'
    ) # type: ignore

    target_empty: PointerProperty(
        name="Target Empty",
        type=bpy.types.Object,
        description="Empty that will receive keyframes (TCP stand-in)",
    ) # type: ignore

    robot_base: PointerProperty(
        name="Robot Base (optional)",
        type=bpy.types.Object,
        description="Robot base object",
    ) # type: ignore

    mm_to_m: FloatProperty(
        name="Millimeters to Meters",
        description="Conversion factor from millimeters to meters",
        default=0.001,
    ) # type: ignore

    global_scale: FloatProperty(
        name="Global Scale",
        description="Scale factor for the entire robot motion",
        default=1.0,
        min=0.001,
    ) # type: ignore

    rot_order: EnumProperty(
        name="Rotation Order",
        description="Order of rotations for Euler angles",
        items=[
            ('XYZ', 'XYZ', ''),
            ('XZY', 'XZY', ''),
            ('YXZ', 'YXZ', ''),
            ('YZX', 'YZX', ''),
            ('ZXY', 'ZXY', ''),
            ('ZYX', 'ZYX', ''),
        ],
        default='XYZ',
    ) # type: ignore

    start_frame: IntProperty(
        name="Start Frame",
        description="Frame to start inserting keyframes",
        default=1,
    ) # type: ignore

    frame_step: IntProperty(
        name="Frame Step",
        description="Number of frames between keyframes",
        default=25,
    ) # type: ignore

    create_action: BoolProperty(
        name="Create New Action",
        default=True,
        description="Create a new action for the imported program",
    ) # type: ignore

class ANIM_OT_import_krl(Operator):
    bl_idname = "anim.import_krl"
    bl_label = "Import KRL Program"
    bl_description = "Import a KRL program and create keyframes for robot motion"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.krl_importer_settings

        if not settings.filepath:
            self.report({'ERROR'}, "No file path specified")
            return {'CANCELLED'}
        
        if not ensure_empty(settings.target_empty):
            self.report({'ERROR'}, "Target object must be an Empty")
            return {'CANCELLED'}
        
        try:
            with open(abspath(settings.filepath), "r", encoding="utf-8", errors="ignore") as f:
                program_text = f.read()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read file: {e}")
            return {'CANCELLED'}
        
        commands = parse_program(program_text)
        if not commands:
            self.report({'ERROR'}, "No valid commands found in the program")
            return {'CANCELLED'}
        
        obj = settings.target_empty

        if settings.create_action:
            obj.animation_data_create()
            action_name = bpy.path.basename(settings.filepath).replace('.src', '').replace('.krl', '').replace('.txt', '')
            obj.animation_data.action = bpy.data.actions.new(name=action_name)

        mm_to_m = settings.mm_to_m * settings.global_scale
        robot_base = settings.robot_base

        frame = settings.start_frame
        pose_count = 0
        pause_count = 0
        last_pose = None

        for command in commands:
            if isinstance(command, dict) and command.get("type") == "WAIT":
                pause_count += 1
                seconds = command["SEC"]
                fps = context.scene.render.fps / context.scene.render.fps_base
                pause_frames = int(seconds * fps)
                print(f"WAIT: {seconds}s at {fps:.2f}fps = {pause_frames} frames (fps={context.scene.render.fps}, fps_base={context.scene.render.fps_base})")

                if last_pose and pause_frames > 0:
                    # Back up to the previous pose's frame (WAIT starts where last pose ended)
                    frame -= settings.frame_step
                    # Add the pause duration from there
                    frame += pause_frames
                    set_empty_pose(obj, last_pose, mm_to_m, settings.rot_order, robot_base)
                    keyframe_pose(obj, frame)
                    # Move to next frame for the next command
                    frame += settings.frame_step
            else:
                pose_count += 1
                set_empty_pose(obj, command, mm_to_m, settings.rot_order, robot_base)
                keyframe_pose(obj, frame)
                last_pose = command
                frame += settings.frame_step

        msg = f"Imported {pose_count} poses"
        if pause_count > 0:
            msg += f" with {pause_count} pauses"
        msg += f" into '{obj.name}'"
        self.report({'INFO'}, msg)
        return {'FINISHED'}
    
class VIEW3D_PT_krl_importer(bpy.types.Panel):
    bl_label = "KRL Importer"
    bl_idname = "VIEW3D_PT_krl_importer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Animation'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.krl_importer_settings

        layout.prop(settings, "filepath")
        layout.prop(settings, "target_empty")
        layout.prop(settings, "robot_base")

        column = layout.column(align=True)
        column.prop(settings, "mm_to_m")
        column.prop(settings, "global_scale")

        layout.prop(settings, "rot_order")

        column = layout.column(align=True)
        column.prop(settings, "start_frame")
        column.prop(settings, "frame_step")

        layout.prop(settings, "create_action")

        layout.operator("anim.import_krl", text="Import KRL Program", icon='IMPORT')

