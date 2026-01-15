import bpy
import re

# MARK: KUKA
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
    r"^\s*WAIT\s+SEC\s+(?P<SEC>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

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
