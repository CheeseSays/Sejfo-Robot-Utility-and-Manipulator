# Interprets KRL a1 a2 a3 a4 a5 a6 commands and converts them to forward kinematics calculations
# using Denavit-Hartenberg parameters for a 6-DOF robotic arm.
# The output is the end-effector position and orientation in Cartesian coordinates, as well as joint angles.
# The animation will be applied to a 3D model hierarchy that looks like this:
# base_link
#  ├── link_1_joint
#  │    └── link_1
#  │         └── link_2_joint
#  │              └── link_2
#  │                   └── link_3_joint
#  │                        └── link_3
#  │                             └── link_4_joint
#  │                                  └── link_4
#  │                                       └── link_5_joint
#  │                                            └── link_5
#  │                                                 └── link_6_joint
#  │                                                      └── link_6

import bpy
import re
import math
import numpy as np
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

# Regular expressions for parsing KRL joint angle commands
# Matches: A1 10.0 A2 20.0 A3 30.0 A4 0 A5 90 A6 0
JOINT_ANGLES_RE = re.compile(
    r"A(?P<joint>[1-6])\s+(?P<angle>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Matches named joint configurations: HOME={A1 0, A2 -90, A3 90, A4 0, A5 90, A6 0}
NAMED_JOINT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*\{(?P<block>[^}]*)\}",
    re.IGNORECASE,
)

# Matches PTP commands with joint angles
PTP_JOINT_RE = re.compile(
    r"^\s*PTP\s+(?P<name>[A-Za-z_]\w*)\b",
    re.IGNORECASE,
)

def parse_joint_angles(text: str) -> dict:
    """Parse joint angles from a KRL command string."""
    angles = {}
    for match in JOINT_ANGLES_RE.finditer(text):
        joint_num = int(match.group("joint"))
        angle = float(match.group("angle"))
        angles[f"A{joint_num}"] = angle
    
    # Ensure all 6 joints have values (default to 0 if not specified)
    for i in range(1, 7):
        angles.setdefault(f"A{i}", 0.0)
    
    return angles

def parse_fk_program(text: str) -> list:
    """Parse a KRL program for joint angle commands."""
    named_configs = {}
    commands = []
    
    lines = text.splitlines()
    for line in lines:
        line_wo_comments = line.split(';')[0].strip()
        if not line_wo_comments:
            continue
        
        # Named joint configuration
        named_match = NAMED_JOINT_RE.match(line_wo_comments)
        if named_match:
            name = named_match.group("name").upper()
            block = named_match.group("block")
            angles = parse_joint_angles(block)
            named_configs[name] = angles
            continue
        
        # PTP with reference to named configuration
        ptp_match = PTP_JOINT_RE.match(line_wo_comments)
        if ptp_match:
            name = ptp_match.group("name").upper()
            if name in named_configs:
                commands.append(named_configs[name])
            continue
        
        # Direct joint angle specification in PTP command
        if re.search(r"\bPTP\b", line_wo_comments, re.IGNORECASE):
            angles = parse_joint_angles(line_wo_comments)
            if any(v != 0.0 for v in angles.values()):
                commands.append(angles)
    
    return commands

def dh_transform(theta, d, a, alpha):
    """
    Create a Denavit-Hartenberg transformation matrix.
    
    Args:
        theta: Joint angle (rotation about z-axis)
        d: Link offset (translation along z-axis)
        a: Link length (translation along x-axis)
        alpha: Link twist (rotation about x-axis)
    
    Returns:
        4x4 transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])

def standard_6dof_dh_params():
    """
    Return standard DH parameters for a typical 6-DOF robot arm.
    These are example values and should be adjusted for specific robot models.
    
    Format: [(d, a, alpha), ...]
    theta is the variable joint angle
    """
    return [
        # Joint 1: (d1, a1, alpha1)
        (0.4, 0.025, np.pi/2),
        # Joint 2: (d2, a2, alpha2)
        (0.0, 0.315, 0.0),
        # Joint 3: (d3, a3, alpha3)
        (0.0, 0.035, np.pi/2),
        # Joint 4: (d4, a4, alpha4)
        (0.365, 0.0, -np.pi/2),
        # Joint 5: (d5, a5, alpha5)
        (0.0, 0.0, np.pi/2),
        # Joint 6: (d6, a6, alpha6)
        (0.095, 0.0, 0.0),
    ]

def forward_kinematics(joint_angles: list, dh_params: list = None):
    """
    Calculate forward kinematics for a 6-DOF robot arm.
    
    Args:
        joint_angles: List of 6 joint angles in radians [a1, a2, a3, a4, a5, a6]
        dh_params: List of DH parameters [(d, a, alpha), ...]. If None, use standard params.
    
    Returns:
        List of 4x4 transformation matrices for each joint (cumulative transformations)
    """
    if dh_params is None:
        dh_params = standard_6dof_dh_params()
    
    transformations = []
    T = np.eye(4)  # Start with identity matrix
    
    for i, (theta, (d, a, alpha)) in enumerate(zip(joint_angles, dh_params)):
        # Calculate transformation for this joint
        T_i = dh_transform(theta, d, a, alpha)
        T = T @ T_i
        transformations.append(T.copy())
    
    return transformations

def matrix_to_blender_transform(matrix):
    """
    Convert a 4x4 numpy transformation matrix to Blender location and rotation.
    
    Returns:
        (location, rotation_matrix) tuple
    """
    import mathutils
    
    # Extract location (translation part)
    location = mathutils.Vector((matrix[0, 3], matrix[1, 3], matrix[2, 3]))
    
    # Extract rotation (3x3 upper-left submatrix)
    rotation_matrix = mathutils.Matrix((
        (matrix[0, 0], matrix[0, 1], matrix[0, 2]),
        (matrix[1, 0], matrix[1, 1], matrix[1, 2]),
        (matrix[2, 0], matrix[2, 1], matrix[2, 2])
    ))
    
    return location, rotation_matrix

def set_joint_rotation(joint_obj, angle_degrees, axis='Z'):
    """Set rotation on a joint object around the specified axis.
    
    Args:
        joint_obj: The joint object to rotate
        angle_degrees: Rotation angle in degrees
        axis: Rotation axis ('X', 'Y', or 'Z')
    """
    if joint_obj and joint_obj.type == 'EMPTY':
        angle_rad = math.radians(angle_degrees)
        joint_obj.rotation_mode = 'XYZ'
        current_rotation = joint_obj.rotation_euler.copy()
        
        # Set rotation based on specified axis
        if axis == 'X':
            current_rotation.x = angle_rad
        elif axis == 'Y':
            current_rotation.y = angle_rad
        else:  # Default to Z
            current_rotation.z = angle_rad
        
        joint_obj.rotation_euler = current_rotation

def keyframe_joint(joint_obj, frame: int):
    """Insert keyframe for joint rotation."""
    if joint_obj:
        joint_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

class FKImporterSettings(PropertyGroup):
    filepath: StringProperty(
        name="File Path",
        description="Path to the KRL program file with joint angle commands (A1-A6)",
        default="",
        subtype='FILE_PATH'
    ) # type: ignore
    
    base_link: PointerProperty(
        name="Base Link",
        type=bpy.types.Object,
        description="Root object of the robot hierarchy (base_link)",
    ) # type: ignore
    
    joint_1: PointerProperty(
        name="Joint 1",
        type=bpy.types.Object,
        description="Link 1 joint object (rotates for A1)",
    ) # type: ignore
    
    joint_2: PointerProperty(
        name="Joint 2",
        type=bpy.types.Object,
        description="Link 2 joint object (rotates for A2)",
    ) # type: ignore
    
    joint_3: PointerProperty(
        name="Joint 3",
        type=bpy.types.Object,
        description="Link 3 joint object (rotates for A3)",
    ) # type: ignore
    
    joint_4: PointerProperty(
        name="Joint 4",
        type=bpy.types.Object,
        description="Link 4 joint object (rotates for A4)",
    ) # type: ignore
    
    joint_5: PointerProperty(
        name="Joint 5",
        type=bpy.types.Object,
        description="Link 5 joint object (rotates for A5)",
    ) # type: ignore
    
    joint_6: PointerProperty(
        name="Joint 6",
        type=bpy.types.Object,
        description="Link 6 joint object (rotates for A6)",
    ) # type: ignore
    
    joint_1_axis: EnumProperty(
        name="J1 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 1"
    ) # type: ignore
    
    joint_2_axis: EnumProperty(
        name="J2 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 2"
    ) # type: ignore
    
    joint_3_axis: EnumProperty(
        name="J3 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 3"
    ) # type: ignore
    
    joint_4_axis: EnumProperty(
        name="J4 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 4"
    ) # type: ignore
    
    joint_5_axis: EnumProperty(
        name="J5 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 5"
    ) # type: ignore
    
    joint_6_axis: EnumProperty(
        name="J6 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 6"
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
        description="Create a new action for each joint",
    ) # type: ignore

class ANIM_OT_import_fk(Operator):
    bl_idname = "anim.import_fk"
    bl_label = "Import FK Program"
    bl_description = "Import a KRL program with joint angles and animate robot using forward kinematics"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        settings = context.scene.fk_importer_settings
        
        if not settings.filepath:
            self.report({'ERROR'}, "No file path specified")
            return {'CANCELLED'}
        
        # Collect joint objects and their rotation axes
        joints = [
            settings.joint_1,
            settings.joint_2,
            settings.joint_3,
            settings.joint_4,
            settings.joint_5,
            settings.joint_6,
        ]
        
        joint_axes = [
            settings.joint_1_axis,
            settings.joint_2_axis,
            settings.joint_3_axis,
            settings.joint_4_axis,
            settings.joint_5_axis,
            settings.joint_6_axis,
        ]
        
        # Verify all joints are set
        missing_joints = []
        for i, joint in enumerate(joints, 1):
            if not joint:
                missing_joints.append(f"Joint {i}")
        
        if missing_joints:
            self.report({'ERROR'}, f"Missing joint assignments: {', '.join(missing_joints)}")
            return {'CANCELLED'}
        
        # Read program file
        try:
            with open(abspath(settings.filepath), "r", encoding="utf-8", errors="ignore") as f:
                program_text = f.read()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read file: {e}")
            return {'CANCELLED'}
        
        # Parse joint angle commands
        commands = parse_fk_program(program_text)
        if not commands:
            self.report({'ERROR'}, "No valid joint angle commands found in the program")
            return {'CANCELLED'}
        
        # Create actions if requested
        if settings.create_action:
            action_name = bpy.path.basename(settings.filepath).replace('.src', '').replace('.krl', '').replace('.txt', '')
            for i, joint in enumerate(joints, 1):
                if joint:
                    joint.animation_data_create()
                    joint.animation_data.action = bpy.data.actions.new(name=f"{action_name}_J{i}")
        
        # Animate joints
        frame = settings.start_frame
        
        for command in commands:
            # Extract joint angles
            angles = [
                command.get('A1', 0.0),
                command.get('A2', 0.0),
                command.get('A3', 0.0),
                command.get('A4', 0.0),
                command.get('A5', 0.0),
                command.get('A6', 0.0),
            ]
            
            # Set joint rotations and keyframe
            for joint_obj, angle, axis in zip(joints, angles, joint_axes):
                set_joint_rotation(joint_obj, angle, axis)
                keyframe_joint(joint_obj, frame)
            
            frame += settings.frame_step
        
        self.report({'INFO'}, f"Imported {len(commands)} joint configurations into robot")
        return {'FINISHED'}

class VIEW3D_PT_fk_importer(bpy.types.Panel):
    bl_label = "FK Importer"
    bl_idname = "VIEW3D_PT_fk_importer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Animation'
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.fk_importer_settings
        
        layout.prop(settings, "filepath")
        
        box = layout.box()
        box.label(text="Robot Joints:")
        box.prop(settings, "base_link")
        
        # Joint 1
        row = box.row()
        row.prop(settings, "joint_1")
        row.prop(settings, "joint_1_axis", text="")
        
        # Joint 2
        row = box.row()
        row.prop(settings, "joint_2")
        row.prop(settings, "joint_2_axis", text="")
        
        # Joint 3
        row = box.row()
        row.prop(settings, "joint_3")
        row.prop(settings, "joint_3_axis", text="")
        
        # Joint 4
        row = box.row()
        row.prop(settings, "joint_4")
        row.prop(settings, "joint_4_axis", text="")
        
        # Joint 5
        row = box.row()
        row.prop(settings, "joint_5")
        row.prop(settings, "joint_5_axis", text="")
        
        # Joint 6
        row = box.row()
        row.prop(settings, "joint_6")
        row.prop(settings, "joint_6_axis", text="")
        
        layout.separator()
        
        column = layout.column(align=True)
        column.prop(settings, "start_frame")
        column.prop(settings, "frame_step")
        
        layout.prop(settings, "create_action")
        
        layout.operator("anim.import_fk", text="Import FK Program", icon='ARMATURE_DATA')
