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