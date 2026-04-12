import numpy as np
from robot.panda import PANDA, JointSpec


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """
    for ref
    
    a     : link length      (metres)
    d     : link offset      (metres)
    alpha : link twist       (radians)
    theta : joint angle      (radians)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,       -st,       0,        a      ],
        [st*ca,     ct*ca,   -sa,      -sa*d   ],
        [st*sa,     ct*sa,    ca,       ca*d   ],
        [0,         0,        0,        1      ],
    ])


def forward_kinematics(q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Compute FK for the Panda at joint configuration q.
    Returns:
    T_tcp : (4, 4) SE(3) transform, TCP in base frame
    T_all : list of 7 (4, 4) transforms, each joint frame in base frame
    """
    assert q.shape == (7,), f"Expected (7,) config, got {q.shape}"

    T = np.eye(4)
    T_all: list[np.ndarray] = []

    for i, (joint, qi) in enumerate(zip(PANDA.JOINTS, q)):
        T_i = _dh_transform(joint.a, joint.d, joint.alpha, qi)
        T = T @ T_i
        T_all.append(T.copy())

    # Attach TCP offset
    T_tcp = T.copy()
    T_tcp[:3, 3] += T_tcp[:3, :3] @ PANDA.TCP_OFFSET

    return T_tcp, T_all


def tcp_position(q: np.ndarray) -> np.ndarray:
    """Return TCP position (3,) in the base frame."""
    T_tcp, _ = forward_kinematics(q)
    return T_tcp[:3, 3]


def tcp_pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (position (3,), rotation_matrix (3, 3)) of TCP."""
    T_tcp, _ = forward_kinematics(q)
    return T_tcp[:3, 3], T_tcp[:3, :3]


def link_positions(q: np.ndarray) -> np.ndarray:
    """
    Return origin position of each joint frame in the base frame.

    Returns
    -------
    positions : (7, 3) array — row i is the origin of frame i
    """
    _, T_all = forward_kinematics(q)
    return np.array([T[:3, 3] for T in T_all])
