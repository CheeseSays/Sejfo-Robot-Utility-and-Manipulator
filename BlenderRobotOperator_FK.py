bl_info = {
    "name": "KUKA Joint Angles -> Empty Animation (Forward Kinematics)",
    "author": "ChatGPT",
    "version": (0, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Animation > KUKA FK Import",
    "description": "Parse KUKA joint angles (A1-A6) and compute TCP position via Forward Kinematics",
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
import numpy as np


# --- KUKA Robot DH Parameters ---
# These are approximate DH parameters for KUKA KR 90 R2700 pro
# Modify these if you have exact parameters for your robot model

KUKA_KR90_DH = {
    # Link 1: Base to A1 rotation
    1: {"a": 500.0, "alpha": math.pi/2, "d": 675.0, "theta_offset": 0.0},
    # Link 2: A1 to A2
    2: {"a": 1150.0, "alpha": 0.0, "d": 0.0, "theta_offset": -math.pi/2},
    # Link 3: A2 to A3
    3: {"a": 100.0, "alpha": math.pi/2, "d": 0.0, "theta_offset": 0.0},
    # Link 4: A3 to A4
    4: {"a": 0.0, "alpha": -math.pi/2, "d": 1200.0, "theta_offset": 0.0},
    # Link 5: A4 to A5
    5: {"a": 0.0, "alpha": math.pi/2, "d": 0.0, "theta_offset": 0.0},
    # Link 6: A5 to A6 (TCP)
    6: {"a": 0.0, "alpha": 0.0, "d": 215.0, "theta_offset": 0.0},
}


# --- Forward Kinematics ---

def dh_transform(a, alpha, d, theta):
    """
    Compute the Denavit-Hartenberg transformation matrix.
    
    Parameters:
    - a: link length
    - alpha: link twist
    - d: link offset
    - theta: joint angle
    
    Returns 4x4 homogeneous transformation matrix
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])


def forward_kinematics(joint_angles, dh_params):
    """
    Compute forward kinematics for a 6-DOF robot.
    
    Parameters:
    - joint_angles: list/tuple of 6 joint angles in radians [A1, A2, A3, A4, A5, A6]
    - dh_params: dictionary with DH parameters for each link
    
    Returns:
    - 4x4 transformation matrix from base to TCP
    """
    T = np.eye(4)
    
    for i in range(1, 7):
        params = dh_params[i]
        theta = joint_angles[i-1] + params["theta_offset"]
        
        T_i = dh_transform(
            a=params["a"],
            alpha=params["alpha"],
            d=params["d"],
            theta=theta
        )
        T = T @ T_i
    
    return T


def forward_kinematics_all_joints(joint_angles, dh_params):
    """
    Compute forward kinematics for all joints of a 6-DOF robot.
    
    Parameters:
    - joint_angles: list/tuple of 6 joint angles in radians [A1, A2, A3, A4, A5, A6]
    - dh_params: dictionary with DH parameters for each link
    
    Returns:
    - list of 6 transformation matrices, one for each joint position
    """
    T = np.eye(4)
    transforms = []
    
    for i in range(1, 7):
        params = dh_params[i]
        theta = joint_angles[i-1] + params["theta_offset"]
        
        T_i = dh_transform(
            a=params["a"],
            alpha=params["alpha"],
            d=params["d"],
            theta=theta
        )
        T = T @ T_i
        transforms.append(T.copy())
    
    return transforms


def extract_position_rotation(T):
    """
    Extract position and rotation from 4x4 transformation matrix.
    
    Returns:
    - position: (x, y, z) in mm
    - rotation_matrix: 3x3 rotation matrix
    """
    position = T[:3, 3]
    rotation = T[:3, :3]
    return position, rotation


def rotation_matrix_to_euler_zyx(R):
    """
    Convert rotation matrix to Euler angles (ZYX convention, intrinsic).
    This is a common convention for KUKA robots.
    
    Returns: (rx, ry, rz) in radians
    """
    # ZYX Euler angles (intrinsic rotations)
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0
    
    return rx, ry, rz


# --- Parsing helpers ---

# Captures joint angle blocks like: {A1 0.00, A2 -90.00, A3 90.00, A4 0.00, A5 0.00, A6 0.00}
JOINT_BLOCK_RE = re.compile(r"\{[^}]*\}", re.IGNORECASE)

# Captures joint angle key-value pairs:
# A1 0.0, A2 -90, A3 90, etc.
JOINT_KV_RE = re.compile(
    r"A(?P<num>[1-6])\s*[:=]?\s*(?P<val>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Captures named joint pose assignments: P1={A1 ..., A2 ...}
NAMED_JOINT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<block>\{[^}]*\})",
    re.IGNORECASE,
)

# Captures PTP commands: PTP P1 or PTP {A1 ...}
PTP_RE = re.compile(
    r"^\s*PTP\s+",
    re.IGNORECASE,
)

# Captures motion commands referencing a named pose: PTP P1
MOTION_REF_RE = re.compile(
    r"^\s*PTP\s+(?P<name>[A-Za-z_]\w*)\b",
    re.IGNORECASE,
)


def parse_joint_block(block: str):
    """
    Parse a joint angle block and return dict with A1-A6.
    Returns None if not all 6 joints are present.
    """
    joints = {}
    for m in JOINT_KV_RE.finditer(block):
        num = int(m.group("num"))
        val = float(m.group("val"))
        joints[f"A{num}"] = val
    
    # Require all 6 joints
    if len(joints) != 6:
        return None
    
    return joints


def parse_kuka_joint_file(text: str):
    """
    Parse KUKA file containing PTP commands with joint angles.
    Returns list of joint angle dicts in motion order.
    Each dict has keys A1, A2, A3, A4, A5, A6 (in degrees).
    """
    named = {}
    motion_joints = []
    
    lines = text.splitlines()
    for line in lines:
        # Strip comments
        line_wo_comment = line.split(";")[0].strip()
        if not line_wo_comment:
            continue
        
        # 1) Named joint pose assignment
        m_named = NAMED_JOINT_RE.match(line_wo_comment)
        if m_named:
            nm = m_named.group("name")
            blk = m_named.group("block")
            joints = parse_joint_block(blk)
            if joints:
                named[nm.upper()] = joints
            continue
        
        # 2) PTP command with inline joint block
        if PTP_RE.match(line_wo_comment):
            blk_m = JOINT_BLOCK_RE.search(line_wo_comment)
            if blk_m:
                joints = parse_joint_block(blk_m.group(0))
                if joints:
                    motion_joints.append(joints)
                    continue
            
            # 3) PTP command referencing named pose
            m_ref = MOTION_REF_RE.match(line_wo_comment)
            if m_ref:
                nm = m_ref.group("name").upper()
                if nm in named:
                    motion_joints.append(named[nm])
                continue
    
    return motion_joints


# --- Blender side ---

def ensure_empty(obj):
    return obj and obj.type == "EMPTY"


def compute_tcp_from_joints(joints_dict, dh_params):
    """
    Convert joint angles dict to TCP position and orientation.
    
    Parameters:
    - joints_dict: dict with A1-A6 in degrees
    - dh_params: DH parameter dictionary
    
    Returns:
    - (x, y, z, rx, ry, rz) where position is in mm and rotation in radians
    """
    # Convert to radians and create array [A1, A2, A3, A4, A5, A6]
    joint_angles = [
        math.radians(joints_dict["A1"]),
        math.radians(joints_dict["A2"]),
        math.radians(joints_dict["A3"]),
        math.radians(joints_dict["A4"]),
        math.radians(joints_dict["A5"]),
        math.radians(joints_dict["A6"]),
    ]
    
    # Compute forward kinematics
    T = forward_kinematics(joint_angles, dh_params)
    
    # Extract position and rotation
    pos, rot_matrix = extract_position_rotation(T)
    x, y, z = pos
    
    # Convert rotation matrix to Euler angles
    rx, ry, rz = rotation_matrix_to_euler_zyx(rot_matrix)
    
    return x, y, z, rx, ry, rz


def set_empty_pose_from_tcp(obj, x, y, z, rx, ry, rz, mm_to_m: float, rot_order: str):
    """
    Set the Empty's position and rotation from TCP coordinates.
    
    Parameters:
    - x, y, z: position in mm
    - rx, ry, rz: rotation in radians
    - mm_to_m: conversion factor
    - rot_order: Euler rotation order
    """
    obj.location = (x * mm_to_m, y * mm_to_m, z * mm_to_m)
    obj.rotation_mode = rot_order
    obj.rotation_euler = (rx, ry, rz)


def keyframe_pose(obj, frame: int):
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)


class KUKAFKImporterSettings(bpy.types.PropertyGroup):
    filepath: StringProperty(
        name="KUKA File",
        subtype="FILE_PATH",
        description="Path to KUKA KRL file with PTP joint commands",
    ) # type: ignore

    target: PointerProperty(
        name="TCP Empty",
        type=bpy.types.Object,
        description="Empty for the TCP (Tool Center Point)",
    ) # type: ignore
    
    axis1: PointerProperty(
        name="Axis 1 Empty",
        type=bpy.types.Object,
        description="Empty for Axis 1 (A1)",
    ) # type: ignore
    
    axis2: PointerProperty(
        name="Axis 2 Empty",
        type=bpy.types.Object,
        description="Empty for Axis 2 (A2)",
    ) # type: ignore
    
    axis3: PointerProperty(
        name="Axis 3 Empty",
        type=bpy.types.Object,
        description="Empty for Axis 3 (A3)",
    ) # type: ignore
    
    axis4: PointerProperty(
        name="Axis 4 Empty",
        type=bpy.types.Object,
        description="Empty for Axis 4 (A4)",
    ) # type: ignore
    
    axis5: PointerProperty(
        name="Axis 5 Empty",
        type=bpy.types.Object,
        description="Empty for Axis 5 (A5)",
    ) # type: ignore
    
    axis6: PointerProperty(
        name="Axis 6 Empty",
        type=bpy.types.Object,
        description="Empty for Axis 6 (A6)",
    ) # type: ignore
    
    auto_create_empties: BoolProperty(
        name="Auto-Create Empties",
        default=True,
        description="Automatically create Empty objects for all axes if they don't exist",
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
        description="Frames to advance per pose",
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
    ) # type: ignore

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

    # DH Parameters - Link 1
    dh_a1: FloatProperty(name="a1", default=500.0, description="Link 1 length (mm)") # type: ignore
    dh_alpha1: FloatProperty(name="α1", default=90.0, description="Link 1 twist (degrees)") # type: ignore
    dh_d1: FloatProperty(name="d1", default=675.0, description="Link 1 offset (mm)") # type: ignore
    
    # DH Parameters - Link 2
    dh_a2: FloatProperty(name="a2", default=1150.0, description="Link 2 length (mm)") # type: ignore
    dh_alpha2: FloatProperty(name="α2", default=0.0, description="Link 2 twist (degrees)") # type: ignore
    dh_d2: FloatProperty(name="d2", default=0.0, description="Link 2 offset (mm)") # type: ignore
    
    # DH Parameters - Link 3
    dh_a3: FloatProperty(name="a3", default=100.0, description="Link 3 length (mm)") # type: ignore
    dh_alpha3: FloatProperty(name="α3", default=90.0, description="Link 3 twist (degrees)") # type: ignore
    dh_d3: FloatProperty(name="d3", default=0.0, description="Link 3 offset (mm)") # type: ignore
    
    # DH Parameters - Link 4
    dh_a4: FloatProperty(name="a4", default=0.0, description="Link 4 length (mm)") # type: ignore
    dh_alpha4: FloatProperty(name="α4", default=-90.0, description="Link 4 twist (degrees)") # type: ignore
    dh_d4: FloatProperty(name="d4", default=1200.0, description="Link 4 offset (mm)") # type: ignore
    
    # DH Parameters - Link 5
    dh_a5: FloatProperty(name="a5", default=0.0, description="Link 5 length (mm)") # type: ignore
    dh_alpha5: FloatProperty(name="α5", default=90.0, description="Link 5 twist (degrees)") # type: ignore
    dh_d5: FloatProperty(name="d5", default=0.0, description="Link 5 offset (mm)") # type: ignore
    
    # DH Parameters - Link 6
    dh_a6: FloatProperty(name="a6", default=0.0, description="Link 6 length (mm)") # type: ignore
    dh_alpha6: FloatProperty(name="α6", default=0.0, description="Link 6 twist (degrees)") # type: ignore
    dh_d6: FloatProperty(name="d6", default=215.0, description="Link 6 offset (mm)") # type: ignore


class ANIM_OT_import_kuka_fk(bpy.types.Operator):
    bl_idname = "anim.import_kuka_fk_to_empty"
    bl_label = "Import KUKA FK -> Empty Animation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.kuka_fk_importer_settings

        if not s.filepath:
            self.report({"ERROR"}, "No file selected.")
            return {"CANCELLED"}

        # Auto-create empties if needed
        if s.auto_create_empties:
            empties = self.create_axis_empties(context, s)
        else:
            empties = {
                "A1": s.axis1,
                "A2": s.axis2,
                "A3": s.axis3,
                "A4": s.axis4,
                "A5": s.axis5,
                "A6": s.axis6,
                "TCP": s.target,
            }
            # Verify all empties exist
            for name, obj in empties.items():
                if not ensure_empty(obj):
                    self.report({"ERROR"}, f"{name} must be an Empty object.")
                    return {"CANCELLED"}

        try:
            with open(bpy.path.abspath(s.filepath), "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read file: {e}")
            return {"CANCELLED"}

        # Build DH parameters from settings (convert alpha from degrees to radians)
        dh_params = {
            1: {"a": s.dh_a1, "alpha": math.radians(s.dh_alpha1), "d": s.dh_d1, "theta_offset": 0.0},
            2: {"a": s.dh_a2, "alpha": math.radians(s.dh_alpha2), "d": s.dh_d2, "theta_offset": -math.pi/2},
            3: {"a": s.dh_a3, "alpha": math.radians(s.dh_alpha3), "d": s.dh_d3, "theta_offset": 0.0},
            4: {"a": s.dh_a4, "alpha": math.radians(s.dh_alpha4), "d": s.dh_d4, "theta_offset": 0.0},
            5: {"a": s.dh_a5, "alpha": math.radians(s.dh_alpha5), "d": s.dh_d5, "theta_offset": 0.0},
            6: {"a": s.dh_a6, "alpha": math.radians(s.dh_alpha6), "d": s.dh_d6, "theta_offset": 0.0},
        }

        joint_poses = parse_kuka_joint_file(text)
        if not joint_poses:
            self.report({"ERROR"}, "No PTP joint poses found. Check file format.")
            return {"CANCELLED"}

        # Clear existing animation if requested
        if s.clear_existing:
            for obj in empties.values():
                if obj:
                    obj.animation_data_clear()

        mm_to_m = s.mm_scale * s.global_scale

        frame = s.start_frame
        for joints in joint_poses:
            try:
                # Compute FK for all joints
                joint_angles = [
                    math.radians(joints["A1"]),
                    math.radians(joints["A2"]),
                    math.radians(joints["A3"]),
                    math.radians(joints["A4"]),
                    math.radians(joints["A5"]),
                    math.radians(joints["A6"]),
                ]
                
                transforms = forward_kinematics_all_joints(joint_angles, dh_params)
                
                # Set pose for each axis
                for i, (axis_name, obj) in enumerate([("A1", empties["A1"]), 
                                                        ("A2", empties["A2"]),
                                                        ("A3", empties["A3"]),
                                                        ("A4", empties["A4"]),
                                                        ("A5", empties["A5"]),
                                                        ("A6", empties["A6"])]):
                    if obj:
                        T = transforms[i]
                        pos, rot_matrix = extract_position_rotation(T)
                        rx, ry, rz = rotation_matrix_to_euler_zyx(rot_matrix)
                        set_empty_pose_from_tcp(obj, pos[0], pos[1], pos[2], rx, ry, rz, 
                                               mm_to_m=mm_to_m, rot_order=s.rot_order)
                        keyframe_pose(obj, frame)
                
                # TCP is the same as A6 position
                if empties["TCP"]:
                    T = transforms[5]
                    pos, rot_matrix = extract_position_rotation(T)
                    rx, ry, rz = rotation_matrix_to_euler_zyx(rot_matrix)
                    set_empty_pose_from_tcp(empties["TCP"], pos[0], pos[1], pos[2], rx, ry, rz,
                                           mm_to_m=mm_to_m, rot_order=s.rot_order)
                    keyframe_pose(empties["TCP"], frame)
                
                frame += s.frame_step
            except Exception as e:
                self.report({"WARNING"}, f"Failed to compute FK for pose: {e}")
                continue

        self.report({"INFO"}, f"Imported {len(joint_poses)} joint poses to robot chain.")
        return {"FINISHED"}
    
    def create_axis_empties(self, context, settings):
        """Create Empty objects for each axis if they don't exist."""
        empties = {}
        
        axis_names = ["A1", "A2", "A3", "A4", "A5", "A6", "TCP"]
        axis_props = [settings.axis1, settings.axis2, settings.axis3, 
                     settings.axis4, settings.axis5, settings.axis6, settings.target]
        
        for i, (name, prop) in enumerate(zip(axis_names, axis_props)):
            if prop and ensure_empty(prop):
                empties[name] = prop
            else:
                # Create new empty
                bpy.ops.object.empty_add(type='PLAIN_AXES', radius=0.1)
                empty = context.active_object
                empty.name = f"KUKA_{name}"
                empties[name] = empty
                
                # Assign to property
                if name == "TCP":
                    settings.target = empty
                elif name == "A1":
                    settings.axis1 = empty
                elif name == "A2":
                    settings.axis2 = empty
                elif name == "A3":
                    settings.axis3 = empty
                elif name == "A4":
                    settings.axis4 = empty
                elif name == "A5":
                    settings.axis5 = empty
                elif name == "A6":
                    settings.axis6 = empty
        
        return empties


class VIEW3D_PT_kuka_fk_import_panel(bpy.types.Panel):
    bl_label = "KUKA FK Import"
    bl_idname = "VIEW3D_PT_kuka_fk_import_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Animation"

    def draw(self, context):
        layout = self.layout
        s = context.scene.kuka_fk_importer_settings

        layout.prop(s, "filepath")
        
        layout.prop(s, "auto_create_empties")
        
        box = layout.box()
        box.label(text="Robot Empties:", icon="EMPTY_AXIS")
        box.prop(s, "axis1")
        box.prop(s, "axis2")
        box.prop(s, "axis3")
        box.prop(s, "axis4")
        box.prop(s, "axis5")
        box.prop(s, "axis6")
        box.prop(s, "target")

        col = layout.column(align=True)
        col.prop(s, "start_frame")
        col.prop(s, "frame_step")

        col = layout.column(align=True)
        col.prop(s, "mm_scale")
        col.prop(s, "global_scale")

        layout.prop(s, "rot_order")
        layout.prop(s, "clear_existing")

        # DH Parameters collapsible section
        box = layout.box()
        box.label(text="DH Parameters (KUKA KR90 R2700)", icon="SETTINGS")
        
        col = box.column(align=True)
        col.label(text="Link 1:")
        col.prop(s, "dh_a1")
        col.prop(s, "dh_alpha1")
        col.prop(s, "dh_d1")
        
        col = box.column(align=True)
        col.label(text="Link 2:")
        col.prop(s, "dh_a2")
        col.prop(s, "dh_alpha2")
        col.prop(s, "dh_d2")
        
        col = box.column(align=True)
        col.label(text="Link 3:")
        col.prop(s, "dh_a3")
        col.prop(s, "dh_alpha3")
        col.prop(s, "dh_d3")
        
        col = box.column(align=True)
        col.label(text="Link 4:")
        col.prop(s, "dh_a4")
        col.prop(s, "dh_alpha4")
        col.prop(s, "dh_d4")
        
        col = box.column(align=True)
        col.label(text="Link 5:")
        col.prop(s, "dh_a5")
        col.prop(s, "dh_alpha5")
        col.prop(s, "dh_d5")
        
        col = box.column(align=True)
        col.label(text="Link 6:")
        col.prop(s, "dh_a6")
        col.prop(s, "dh_alpha6")
        col.prop(s, "dh_d6")

        layout.operator(ANIM_OT_import_kuka_fk.bl_idname, icon="IMPORT")


classes = (
    KUKAFKImporterSettings,
    ANIM_OT_import_kuka_fk,
    VIEW3D_PT_kuka_fk_import_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.kuka_fk_importer_settings = PointerProperty(type=KUKAFKImporterSettings)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.kuka_fk_importer_settings


if __name__ == "__main__":
    register()
