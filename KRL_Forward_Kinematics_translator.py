# Interprets KRL a1 a2 a3 a4 a5 a6 commands and converts them to forward kinematics calculations
# using Denavit-Hartenberg parameters for a 6-DOF robotic arm.
# The output is the end-effector position and orientation in Cartesian coordinates, as well as joint angles.
# The animation will be applied to an ARMATURE with bones representing the robot joints:
# Robot_Armature (armature object)
#  └── Bone hierarchy:
#       Joint_1 (bone) -> rotates for A1
#       └── Joint_2 (bone) -> rotates for A2
#            └── Joint_3 (bone) -> rotates for A3
#                 └── Joint_4 (bone) -> rotates for A4
#                      └── Joint_5 (bone) -> rotates for A5
#                           └── Joint_6 (bone) -> rotates for A6

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

# Matches velocity commands: BAS (#VEL_PTP,100)
VEL_PTP_RE = re.compile(
    r"BAS\s*\(\s*#VEL_PTP\s*,\s*(?P<velocity>\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)

# Matches acceleration commands: BAS (#ACC_PTP,20)
ACC_PTP_RE = re.compile(
    r"BAS\s*\(\s*#ACC_PTP\s*,\s*(?P<acceleration>\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)

# Matches Cartesian path velocity: $VEL.CP=0.2
VEL_CP_RE = re.compile(
    r"\$VEL\.CP\s*=\s*(?P<velocity>\d+(?:\.\d+)?)",
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

def parse_fk_program(text: str) -> tuple:
    """Parse a KRL program for joint angle commands and motion parameters.
    
    Returns:
        tuple: (commands, velocity, acceleration) where:
            - commands: list of joint angle dictionaries
            - velocity: PTP velocity percentage (default 100)
            - acceleration: PTP acceleration percentage (default 100)
    """
    named_configs = {}
    commands = []
    velocity = 100.0  # Default PTP velocity percentage
    acceleration = 100.0  # Default PTP acceleration percentage
    
    lines = text.splitlines()
    for line in lines:
        line_wo_comments = line.split(';')[0].strip()
        if not line_wo_comments:
            continue
        
        # Check for velocity setting
        vel_match = VEL_PTP_RE.search(line_wo_comments)
        if vel_match:
            velocity = float(vel_match.group("velocity"))
            continue
        
        # Check for acceleration setting
        acc_match = ACC_PTP_RE.search(line_wo_comments)
        if acc_match:
            acceleration = float(acc_match.group("acceleration"))
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
            # Accept all PTP commands, including those with all zeros (e.g., home position)
            commands.append(angles)
    
    return commands, velocity, acceleration

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

def dh_transform_with_axis(theta, d, a, alpha, rotation_axis='Z'):
    """Create a DH transformation matrix with rotation around a specified axis.
    
    Args:
        theta: Joint angle
        d: Link offset (translation along z-axis)
        a: Link length (translation along x-axis)
        alpha: Link twist (rotation about x-axis)
        rotation_axis: 'X', 'Y', or 'Z' - axis of joint rotation
    
    Returns:
        4x4 transformation matrix
    """
    # Standard DH assumes Z-axis rotation
    if rotation_axis == 'Z':
        return dh_transform(theta, d, a, alpha)
    
    # For X or Y axis rotation, we need to modify the transformation
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    if rotation_axis == 'X':
        # Rotation around X-axis instead of Z
        # First do translation, then rotate around X, then apply twist
        T_trans = np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1]
        ])
        T_rot_x = np.array([
            [1,  0,   0,  0],
            [0,  ct, -st, 0],
            [0,  st,  ct, 0],
            [0,  0,   0,  1]
        ])
        T_twist = np.array([
            [1,  0,   0,  0],
            [0,  ca, -sa, 0],
            [0,  sa,  ca, 0],
            [0,  0,   0,  1]
        ])
        return T_trans @ T_rot_x @ T_twist
    
    elif rotation_axis == 'Y':
        # Rotation around Y-axis instead of Z
        T_trans = np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1]
        ])
        T_rot_y = np.array([
            [ct,  0, st, 0],
            [0,   1, 0,  0],
            [-st, 0, ct, 0],
            [0,   0, 0,  1]
        ])
        T_twist = np.array([
            [1,  0,   0,  0],
            [0,  ca, -sa, 0],
            [0,  sa,  ca, 0],
            [0,  0,   0,  1]
        ])
        return T_trans @ T_rot_y @ T_twist
    
    return dh_transform(theta, d, a, alpha)

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

def forward_kinematics(joint_angles: list, dh_params: list = None, rotation_axes: list = None):
    """
    Calculate forward kinematics for a 6-DOF robot arm.
    
    Args:
        joint_angles: List of 6 joint angles in radians [a1, a2, a3, a4, a5, a6]
        dh_params: List of DH parameters [(d, a, alpha), ...]. If None, use standard params.
        rotation_axes: List of rotation axes ['X'/'Y'/'Z', ...]. If None, use 'Z' for all.
    
    Returns:
        List of 4x4 transformation matrices for each joint (cumulative transformations)
    """
    if dh_params is None:
        dh_params = standard_6dof_dh_params()
    
    if rotation_axes is None:
        rotation_axes = ['Z'] * 6
    
    transformations = []
    T = np.eye(4)  # Start with identity matrix
    
    for i, (theta, (d, a, alpha), axis) in enumerate(zip(joint_angles, dh_params, rotation_axes)):
        # Calculate transformation for this joint with specified rotation axis
        T_i = dh_transform_with_axis(theta, d, a, alpha, axis)
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

def calculate_dh_parameters_from_bones(armature_obj, bone_names, joint_axes):
    """Calculate DH parameters from armature bones.
    
    Args:
        armature_obj: The armature object
        bone_names: List of 6 bone names
        joint_axes: List of rotation axes for each joint
    
    Returns:
        List of (d, a, alpha) tuples for each joint
    """
    import mathutils
    
    if not armature_obj or armature_obj.type != 'ARMATURE':
        return standard_6dof_dh_params()
    
    dh_params = []
    armature = armature_obj.data
    
    # Start from armature origin
    prev_pos = armature_obj.matrix_world.translation
    prev_z = mathutils.Vector((0, 0, 1))  # Initial Z axis
    
    for i, (bone_name, axis) in enumerate(zip(bone_names, joint_axes)):
        if not bone_name or bone_name not in armature.bones:
            # Use default parameters if bone is not set
            dh_params.append((0.0, 0.0, 0.0))
            continue
        
        bone = armature.bones[bone_name]
        
        # Get bone head position in world space
        bone_head_world = armature_obj.matrix_world @ bone.head_local
        
        # Calculate link offset (d) - distance along previous Z axis
        offset_vec = bone_head_world - prev_pos
        d = offset_vec.dot(prev_z)
        
        # Calculate link length (a) - distance in XY plane perpendicular to Z
        z_component = prev_z * d
        xy_component = offset_vec - z_component
        a = xy_component.length
        
        # Calculate link twist (alpha) - angle between Z axes
        # Get bone's local Z axis in world space based on rotation axis setting
        bone_matrix = (armature_obj.matrix_world @ bone.matrix_local).to_3x3()
        if axis == 'X':
            current_z = bone_matrix @ mathutils.Vector((1, 0, 0))
        elif axis == 'Y':
            current_z = bone_matrix @ mathutils.Vector((0, 1, 0))
        else:  # Z
            current_z = bone_matrix @ mathutils.Vector((0, 0, 1))
        
        # Alpha is the angle between prev_z and current_z
        cos_alpha = prev_z.dot(current_z)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))  # Clamp to valid range
        alpha = math.acos(cos_alpha)
        
        dh_params.append((d, a, alpha))
        
        # Update for next iteration - use bone tail for next position
        prev_pos = armature_obj.matrix_world @ bone.tail_local
        prev_z = current_z
    
    return dh_params

def apply_fk_to_joint(joint_obj, transformation_matrix, base_link=None):
    """Apply a forward kinematics transformation matrix to a joint object.
    
    Args:
        joint_obj: The joint object to transform
        transformation_matrix: 4x4 numpy transformation matrix from FK
        base_link: Optional base link for relative positioning
    """
    import mathutils
    
    if not joint_obj:
        return
    
    # Convert numpy matrix to Blender matrix
    blender_matrix = mathutils.Matrix((
        (transformation_matrix[0, 0], transformation_matrix[0, 1], transformation_matrix[0, 2], transformation_matrix[0, 3]),
        (transformation_matrix[1, 0], transformation_matrix[1, 1], transformation_matrix[1, 2], transformation_matrix[1, 3]),
        (transformation_matrix[2, 0], transformation_matrix[2, 1], transformation_matrix[2, 2], transformation_matrix[2, 3]),
        (transformation_matrix[3, 0], transformation_matrix[3, 1], transformation_matrix[3, 2], transformation_matrix[3, 3])
    ))
    
    # Apply relative to base_link if provided
    if base_link:
        world_matrix = base_link.matrix_world @ blender_matrix
    else:
        world_matrix = blender_matrix
    
    # Apply to joint object
    joint_obj.matrix_world = world_matrix

def set_bone_rotation(pose_bone, angle_degrees, axis='Z', invert=False):
    """Set rotation on a pose bone around the specified axis.
    
    Args:
        pose_bone: The pose bone to rotate
        angle_degrees: Rotation angle in degrees
        axis: Rotation axis ('X', 'Y', or 'Z')
        invert: If True, negate the rotation angle
    """
    if not pose_bone:
        return
    
    # Apply inversion if needed
    if invert:
        angle_degrees = -angle_degrees
    
    angle_rad = math.radians(angle_degrees)
    pose_bone.rotation_mode = 'XYZ'
    
    # Set rotation based on specified axis (in local space)
    if axis == 'X':
        pose_bone.rotation_euler[0] = angle_rad
    elif axis == 'Y':
        pose_bone.rotation_euler[1] = angle_rad
    else:  # Default to Z
        pose_bone.rotation_euler[2] = angle_rad

def keyframe_bone(pose_bone, frame: int):
    """Insert keyframe for bone rotation."""
    if pose_bone:
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)

def calculate_dynamic_frame_step(prev_angles, curr_angles, velocity, acceleration, base_fps, velocity_scale):
    """Calculate dynamic frame step based on angular displacement, velocity, and acceleration.
    
    Args:
        prev_angles: Previous joint angles in degrees [A1, A2, A3, A4, A5, A6]
        curr_angles: Current joint angles in degrees [A1, A2, A3, A4, A5, A6]
        velocity: PTP velocity percentage (0-100)
        acceleration: PTP acceleration percentage (0-100)
        base_fps: Base frames per second for the animation
        velocity_scale: Scale factor for velocity calculation
    
    Returns:
        Number of frames for this movement segment
    """
    # Calculate the maximum angular displacement across all joints
    max_displacement = 0.0
    for prev, curr in zip(prev_angles, curr_angles):
        displacement = abs(curr - prev)
        max_displacement = max(max_displacement, displacement)
    
    # If no movement, return minimum frame step
    if max_displacement < 0.001:
        return 1
    
    # Base time calculation (assuming max velocity is around 150 deg/s for industrial robots)
    # Scale velocity: 100% = 1.0, 50% = 0.5
    velocity_factor = velocity / 100.0
    
    # Estimate time based on displacement and velocity
    # Using a reasonable max angular velocity for industrial robots
    max_angular_velocity = 150.0  # degrees per second at 100% velocity
    actual_velocity = max_angular_velocity * velocity_factor
    
    # Calculate movement time in seconds
    # With acceleration, we need to account for ramp-up/ramp-down
    # Simplified trapezoidal velocity profile
    accel_factor = acceleration / 100.0
    
    # Assuming acceleration time is ~30% of total movement time at 100%
    # Lower acceleration means longer ramp times
    accel_time_ratio = 0.3 / accel_factor
    
    # Time = displacement / velocity, adjusted for acceleration profile
    if max_displacement < (actual_velocity * accel_time_ratio):
        # Short movement - dominated by acceleration
        movement_time = 2 * math.sqrt(max_displacement / (actual_velocity * accel_factor))
    else:
        # Long movement - trapezoidal profile
        const_vel_displacement = max_displacement - (actual_velocity * accel_time_ratio)
        movement_time = accel_time_ratio + (const_vel_displacement / actual_velocity)
    
    # Apply velocity scale (user adjustment)
    movement_time *= velocity_scale
    
    # Convert to frames
    frames = max(1, int(movement_time * base_fps))
    
    return frames

class FKImporterSettings(PropertyGroup):
    filepath: StringProperty(
        name="File Path",
        description="Path to the KRL program file with joint angle commands (A1-A6)",
        default="",
        subtype='FILE_PATH'
    ) # type: ignore
    
    armature: PointerProperty(
        name="Robot Armature",
        type=bpy.types.Object,
        description="Armature object containing the robot bones",
    ) # type: ignore
    
    bone_1: StringProperty(
        name="Joint 1 Bone",
        description="Name of bone for Joint 1 (A1)",
        default="Joint_1"
    ) # type: ignore
    
    bone_2: StringProperty(
        name="Joint 2 Bone",
        description="Name of bone for Joint 2 (A2)",
        default="Joint_2"
    ) # type: ignore
    
    bone_3: StringProperty(
        name="Joint 3 Bone",
        description="Name of bone for Joint 3 (A3)",
        default="Joint_3"
    ) # type: ignore
    
    bone_4: StringProperty(
        name="Joint 4 Bone",
        description="Name of bone for Joint 4 (A4)",
        default="Joint_4"
    ) # type: ignore
    
    bone_5: StringProperty(
        name="Joint 5 Bone",
        description="Name of bone for Joint 5 (A5)",
        default="Joint_5"
    ) # type: ignore
    
    bone_6: StringProperty(
        name="Joint 6 Bone",
        description="Name of bone for Joint 6 (A6)",
        default="Joint_6"
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
    
    invert_joint_1: BoolProperty(
        name="Invert J1",
        default=False,
        description="Invert rotation direction for Joint 1"
    ) # type: ignore
    
    invert_joint_2: BoolProperty(
        name="Invert J2",
        default=False,
        description="Invert rotation direction for Joint 2"
    ) # type: ignore
    
    invert_joint_3: BoolProperty(
        name="Invert J3",
        default=False,
        description="Invert rotation direction for Joint 3"
    ) # type: ignore
    
    invert_joint_4: BoolProperty(
        name="Invert J4",
        default=False,
        description="Invert rotation direction for Joint 4"
    ) # type: ignore
    
    invert_joint_5: BoolProperty(
        name="Invert J5",
        default=False,
        description="Invert rotation direction for Joint 5"
    ) # type: ignore
    
    invert_joint_6: BoolProperty(
        name="Invert J6",
        default=False,
        description="Invert rotation direction for Joint 6"
    ) # type: ignore
    
    start_frame: IntProperty(
        name="Start Frame",
        description="Frame to start inserting keyframes",
        default=1,
    ) # type: ignore
    
    frame_step: IntProperty(
        name="Frame Step",
        description="Number of frames between keyframes (Fixed mode only)",
        default=25,
    ) # type: ignore
    
    frame_step_mode: EnumProperty(
        name="Frame Step Mode",
        items=[
            ('FIXED', 'Fixed', 'Use fixed frame step value for all movements'),
            ('DYNAMIC', 'Dynamic', 'Calculate frame steps based on velocity, acceleration, and angular displacement'),
        ],
        default='FIXED',
        description="Method for calculating frames between keyframes"
    ) # type: ignore
    
    base_fps: FloatProperty(
        name="Base FPS",
        description="Base frames per second for dynamic frame calculation (typically 24, 30, or 60)",
        default=30.0,
        min=1.0,
        max=120.0,
    ) # type: ignore
    
    velocity_scale: FloatProperty(
        name="Velocity Scale",
        description="Scale factor for velocity-based frame calculation (higher = slower motion)",
        default=1.0,
        min=0.1,
        max=10.0,
    ) # type: ignore
    
    create_action: BoolProperty(
        name="Create New Action",
        default=True,
        description="Create a new action for each joint",
    ) # type: ignore
    
    # DH Parameters (calculated from scene)
    dh_params_calculated: BoolProperty(
        name="DH Params Calculated",
        default=False,
        description="Whether DH parameters have been calculated from the scene",
    ) # type: ignore
    
    show_dh_params: BoolProperty(
        name="Show DH Parameters",
        default=False,
        description="Show the calculated DH parameters",
    ) # type: ignore
    
    tcp_object: PointerProperty(
        name="TCP Object (Optional)",
        type=bpy.types.Object,
        description="Object to animate with calculated TCP position from forward kinematics",
    ) # type: ignore
    
    animate_tcp: BoolProperty(
        name="Animate TCP",
        default=False,
        description="Calculate and animate TCP position using forward kinematics",
    ) # type: ignore
    
    # Store DH parameters as strings (Blender properties can't store tuples/lists directly)
    dh_param_d1: FloatProperty(name="d1", default=0.0) # type: ignore
    dh_param_a1: FloatProperty(name="a1", default=0.0) # type: ignore
    dh_param_alpha1: FloatProperty(name="alpha1", default=0.0) # type: ignore
    dh_param_d2: FloatProperty(name="d2", default=0.0) # type: ignore
    dh_param_a2: FloatProperty(name="a2", default=0.0) # type: ignore
    dh_param_alpha2: FloatProperty(name="alpha2", default=0.0) # type: ignore
    dh_param_d3: FloatProperty(name="d3", default=0.0) # type: ignore
    dh_param_a3: FloatProperty(name="a3", default=0.0) # type: ignore
    dh_param_alpha3: FloatProperty(name="alpha3", default=0.0) # type: ignore
    dh_param_d4: FloatProperty(name="d4", default=0.0) # type: ignore
    dh_param_a4: FloatProperty(name="a4", default=0.0) # type: ignore
    dh_param_alpha4: FloatProperty(name="alpha4", default=0.0) # type: ignore
    dh_param_d5: FloatProperty(name="d5", default=0.0) # type: ignore
    dh_param_a5: FloatProperty(name="a5", default=0.0) # type: ignore
    dh_param_alpha5: FloatProperty(name="alpha5", default=0.0) # type: ignore
    dh_param_d6: FloatProperty(name="d6", default=0.0) # type: ignore
    dh_param_a6: FloatProperty(name="a6", default=0.0) # type: ignore
    dh_param_alpha6: FloatProperty(name="alpha6", default=0.0) # type: ignore

class ANIM_OT_calculate_dh(Operator):
    bl_idname = "anim.calculate_dh"
    bl_label = "Calculate DH Parameters"
    bl_description = "Calculate Denavit-Hartenberg parameters from the armature bones"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        settings = context.scene.fk_importer_settings
        
        if not settings.armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}
        
        if settings.armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Selected object is not an armature")
            return {'CANCELLED'}
        
        # Collect bone names and axes
        bone_names = [
            settings.bone_1,
            settings.bone_2,
            settings.bone_3,
            settings.bone_4,
            settings.bone_5,
            settings.bone_6,
        ]
        
        joint_axes = [
            settings.joint_1_axis,
            settings.joint_2_axis,
            settings.joint_3_axis,
            settings.joint_4_axis,
            settings.joint_5_axis,
            settings.joint_6_axis,
        ]
        
        # Check if all bones exist
        armature = settings.armature.data
        missing_bones = []
        for i, bone_name in enumerate(bone_names, 1):
            if not bone_name or bone_name not in armature.bones:
                missing_bones.append(f"Joint {i} ({bone_name})")
        
        if missing_bones:
            self.report({'WARNING'}, f"Some bones not found: {', '.join(missing_bones)}. Using defaults for missing bones.")
        
        # Calculate DH parameters
        dh_params = calculate_dh_parameters_from_bones(settings.armature, bone_names, joint_axes)
        
        # Store them in settings
        for i, (d, a, alpha) in enumerate(dh_params, 1):
            setattr(settings, f"dh_param_d{i}", d)
            setattr(settings, f"dh_param_a{i}", a)
            setattr(settings, f"dh_param_alpha{i}", alpha)
        
        settings.dh_params_calculated = True
        
        # Display parameters
        params_str = "DH Parameters (d, a, alpha):\n"
        for i, (d, a, alpha) in enumerate(dh_params, 1):
            params_str += f"Joint {i}: d={d:.4f}m, a={a:.4f}m, α={math.degrees(alpha):.2f}°\n"
        
        self.report({'INFO'}, f"DH parameters calculated from armature bones")
        print(params_str)
        
        return {'FINISHED'}

class ANIM_OT_import_fk(Operator):
    bl_idname = "anim.import_fk"
    bl_label = "Import FK Program"
    bl_description = "Import a KRL program with joint angles and animate robot armature using forward kinematics"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        settings = context.scene.fk_importer_settings
        
        if not settings.filepath:
            self.report({'ERROR'}, "No file path specified")
            return {'CANCELLED'}
        
        if not settings.armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}
        
        if settings.armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Selected object is not an armature")
            return {'CANCELLED'}
        
        # Collect bone names and their rotation axes
        bone_names = [
            settings.bone_1,
            settings.bone_2,
            settings.bone_3,
            settings.bone_4,
            settings.bone_5,
            settings.bone_6,
        ]
        
        joint_axes = [
            settings.joint_1_axis,
            settings.joint_2_axis,
            settings.joint_3_axis,
            settings.joint_4_axis,
            settings.joint_5_axis,
            settings.joint_6_axis,
        ]
        
        invert_flags = [
            settings.invert_joint_1,
            settings.invert_joint_2,
            settings.invert_joint_3,
            settings.invert_joint_4,
            settings.invert_joint_5,
            settings.invert_joint_6,
        ]
        
        # Verify all bones exist in armature
        armature = settings.armature.data
        pose_bones = settings.armature.pose.bones
        missing_bones = []
        for i, bone_name in enumerate(bone_names, 1):
            if not bone_name or bone_name not in armature.bones:
                missing_bones.append(f"Joint {i} ({bone_name})")
        
        if missing_bones:
            self.report({'ERROR'}, f"Missing bones in armature: {', '.join(missing_bones)}")
            return {'CANCELLED'}
        
        # Read program file
        try:
            with open(abspath(settings.filepath), "r", encoding="utf-8", errors="ignore") as f:
                program_text = f.read()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read file: {e}")
            return {'CANCELLED'}
        
        # Parse joint angle commands
        parse_result = parse_fk_program(program_text)
        commands, velocity, acceleration = parse_result
        if not commands:
            self.report({'ERROR'}, "No valid joint angle commands found in the program")
            return {'CANCELLED'}
        
        # Log velocity and acceleration settings
        print(f"\n=== Motion Parameters ===")
        print(f"PTP Velocity: {velocity}%")
        print(f"PTP Acceleration: {acceleration}%")
        print(f"========================\n")
        
        # Calculate DH parameters from the armature
        dh_params = calculate_dh_parameters_from_bones(settings.armature, bone_names, joint_axes)
        
        # Store them in settings
        for i, (d, a, alpha) in enumerate(dh_params, 1):
            setattr(settings, f"dh_param_d{i}", d)
            setattr(settings, f"dh_param_a{i}", a)
            setattr(settings, f"dh_param_alpha{i}", alpha)
        
        settings.dh_params_calculated = True
        
        # Display calculated parameters
        print("\n=== Calculated DH Parameters ===")
        for i, (d, a, alpha) in enumerate(dh_params, 1):
            print(f"Joint {i}: d={d:.4f}m, a={a:.4f}m, α={math.degrees(alpha):.2f}°")
        print("================================\n")
        
        # Setup TCP animation if requested
        tcp_obj = None
        if settings.animate_tcp and settings.tcp_object:
            tcp_obj = settings.tcp_object
            if settings.create_action:
                tcp_obj.animation_data_create()
                action_name = bpy.path.basename(settings.filepath).replace('.src', '').replace('.krl', '').replace('.txt', '')
                tcp_obj.animation_data.action = bpy.data.actions.new(name=f"{action_name}_TCP")
        
        # Create action for armature if requested
        if settings.create_action:
            settings.armature.animation_data_create()
            action_name = bpy.path.basename(settings.filepath).replace('.src', '').replace('.krl', '').replace('.txt', '')
            settings.armature.animation_data.action = bpy.data.actions.new(name=f"{action_name}_Armature")
        
        # Animate bones
        frame = settings.start_frame
        prev_angles = None
        
        for cmd_idx, command in enumerate(commands):
            # Extract joint angles
            angles = [
                command.get('A1', 0.0),
                command.get('A2', 0.0),
                command.get('A3', 0.0),
                command.get('A4', 0.0),
                command.get('A5', 0.0),
                command.get('A6', 0.0),
            ]
            
            # Apply rotations directly to pose bones
            for i, (bone_name, angle, axis, invert) in enumerate(zip(bone_names, angles, joint_axes, invert_flags)):
                if bone_name in pose_bones:
                    pose_bone = pose_bones[bone_name]
                    set_bone_rotation(pose_bone, angle, axis, invert)
                    keyframe_bone(pose_bone, frame)
            
            # Calculate and animate TCP position using forward kinematics if requested
            if tcp_obj and settings.animate_tcp:
                # Convert angles to radians for FK calculation
                angles_rad = [math.radians(a) for a in angles]
                transformations = forward_kinematics(angles_rad, dh_params, joint_axes)
                
                if transformations:
                    final_transform = transformations[-1]
                    # Apply FK transformation to TCP
                    apply_fk_to_joint(tcp_obj, final_transform, settings.armature)
                    # Keyframe TCP
                    tcp_obj.keyframe_insert(data_path="location", frame=frame)
                    tcp_obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            
            # Calculate frame step for next position
            if cmd_idx < len(commands) - 1:  # Not the last command
                if settings.frame_step_mode == 'DYNAMIC':
                    # Calculate dynamic frame step based on motion
                    if prev_angles is None:
                        prev_angles = angles
                    
                    frame_delta = calculate_dynamic_frame_step(
                        angles,  # Current becomes previous for next iteration
                        [commands[cmd_idx + 1].get(f'A{i}', 0.0) for i in range(1, 7)],  # Peek at next
                        velocity, 
                        acceleration, 
                        settings.base_fps, 
                        settings.velocity_scale
                    )
                    frame += frame_delta
                    print(f"Command {cmd_idx + 1} -> {cmd_idx + 2}: Dynamic frame step = {frame_delta} (next frame {frame})")
                else:
                    # Fixed frame step mode
                    frame += settings.frame_step
            
            # Update previous angles for reference
            prev_angles = angles
        
        self.report({'INFO'}, f"Imported {len(commands)} joint configurations into armature")
        return {'FINISHED'}

class VIEW3D_PT_fk_importer(bpy.types.Panel):
    bl_label = "FK Importer (Armature)"
    bl_idname = "VIEW3D_PT_fk_importer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Animation'
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.fk_importer_settings
        
        layout.prop(settings, "filepath")
        
        box = layout.box()
        box.label(text="Robot Armature:", icon='ARMATURE_DATA')
        box.prop(settings, "armature")
        
        if settings.armature and settings.armature.type == 'ARMATURE':
            box.label(text="Bone Names, Axes & Invert:")
            
            # Joint 1
            row = box.row()
            row.prop(settings, "bone_1", text="J1")
            row.prop(settings, "joint_1_axis", text="")
            row.prop(settings, "invert_joint_1", text="", icon='ARROW_LEFTRIGHT')
            
            # Joint 2
            row = box.row()
            row.prop(settings, "bone_2", text="J2")
            row.prop(settings, "joint_2_axis", text="")
            row.prop(settings, "invert_joint_2", text="", icon='ARROW_LEFTRIGHT')
            
            # Joint 3
            row = box.row()
            row.prop(settings, "bone_3", text="J3")
            row.prop(settings, "joint_3_axis", text="")
            row.prop(settings, "invert_joint_3", text="", icon='ARROW_LEFTRIGHT')
            
            # Joint 4
            row = box.row()
            row.prop(settings, "bone_4", text="J4")
            row.prop(settings, "joint_4_axis", text="")
            row.prop(settings, "invert_joint_4", text="", icon='ARROW_LEFTRIGHT')
            
            # Joint 5
            row = box.row()
            row.prop(settings, "bone_5", text="J5")
            row.prop(settings, "joint_5_axis", text="")
            row.prop(settings, "invert_joint_5", text="", icon='ARROW_LEFTRIGHT')
            
            # Joint 6
            row = box.row()
            row.prop(settings, "bone_6", text="J6")
            row.prop(settings, "joint_6_axis", text="")
            row.prop(settings, "invert_joint_6", text="", icon='ARROW_LEFTRIGHT')
        elif settings.armature:
            box.label(text="⚠ Selected object is not an armature", icon='ERROR')
        
        layout.separator()
        
        # Frame stepping configuration
        box = layout.box()
        box.label(text="Frame Stepping:", icon='TIME')
        box.prop(settings, "frame_step_mode", text="Mode")
        
        column = box.column(align=True)
        column.prop(settings, "start_frame")
        
        if settings.frame_step_mode == 'FIXED':
            column.prop(settings, "frame_step")
            box.label(text="Fixed frames between keyframes", icon='INFO')
        else:  # DYNAMIC
            column.prop(settings, "base_fps")
            column.prop(settings, "velocity_scale")
            box.label(text="Frames calculated from velocity/acceleration", icon='INFO')
        
        layout.prop(settings, "create_action")
        
        layout.separator()
        
        # DH Parameters section
        box = layout.box()
        box.label(text="DH Parameters:", icon='SETTINGS')
        box.operator("anim.calculate_dh", text="Calculate DH from Scene", icon='CON_KINEMATIC')
        
        if settings.dh_params_calculated:
            box.label(text="✓ Parameters calculated", icon='CHECKMARK')
            box.label(text="(Check console for values)")
        
        layout.separator()
        
        # TCP Animation section
        box = layout.box()
        box.label(text="TCP Animation (Optional):", icon='EMPTY_AXIS')
        box.prop(settings, "animate_tcp")
        if settings.animate_tcp:
            box.prop(settings, "tcp_object")
            if not settings.tcp_object:
                box.label(text="⚠ Select TCP object", icon='ERROR')
        
        layout.separator()
        
        layout.operator("anim.import_fk", text="Import FK Program", icon='ARMATURE_DATA')
