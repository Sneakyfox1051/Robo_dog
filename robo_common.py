"""Shared PyBullet helpers for Laikago (Z-up URDF) and Minitaur demos."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pybullet as p
import pybullet_data

LAIKAGO_URDF = "laikago/laikago_toes_zup.urdf"
MINITAUR_URDF = "quadruped/minitaur.urdf"

LAIKAGO_JOINT_NAMES: tuple[str, ...] = (
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint",
    "FR_lower_leg_2_upper_leg_joint",
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint",
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    "RL_hip_motor_2_chassis_joint",
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint",
)

# SDK space: (abduction, hip, knee) × 4 legs
INIT_SDK: tuple[float, ...] = (0.0, 0.67, -1.25) * 4
_SDK_DIRECTIONS = np.array((-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1), dtype=np.float64)
_SDK_OFFSETS = np.array((0.0, -0.6, 0.66) * 4, dtype=np.float64)

LAIKAGO_TOE_LINK_NAMES: tuple[str, ...] = ("toeFR", "toeFL", "toeRR", "toeRL")

MINITAUR_MOTOR_NAMES: tuple[str, ...] = (
    "motor_front_rightR_joint",
    "motor_front_rightL_joint",
    "motor_front_leftL_joint",
    "motor_front_leftR_joint",
    "motor_back_rightR_joint",
    "motor_back_rightL_joint",
    "motor_back_leftL_joint",
    "motor_back_leftR_joint",
)


def sdk_to_urdf(sdk: np.ndarray | Sequence[float]) -> np.ndarray:
    s = np.asarray(sdk, dtype=np.float64)
    return (s + _SDK_OFFSETS) * _SDK_DIRECTIONS


def laikago_joint_indices(robot: int) -> list[int]:
    name_to_index: dict[str, int] = {}
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name_to_index[info[1].decode("utf-8")] = i
    missing = [n for n in LAIKAGO_JOINT_NAMES if n not in name_to_index]
    if missing:
        raise RuntimeError(f"Laikago URDF missing joints: {missing}")
    return [name_to_index[n] for n in LAIKAGO_JOINT_NAMES]


def link_index_from_name(robot: int, link_name: str) -> int:
    bi = p.getBodyInfo(robot)
    if bi[0].decode("utf-8") == link_name:
        return -1
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        if info[12].decode("utf-8") == link_name:
            return int(info[16])
    raise RuntimeError(f"Link not found: {link_name}")


def connect(gui: bool = False) -> int:
    return p.connect(p.GUI if gui else p.DIRECT)


def setup_scene(plane: bool = True) -> None:
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    if plane:
        p.loadURDF("plane.urdf")


def load_laikago(
    start_pos: Sequence[float] | None = None,
    start_rpy: Sequence[float] | None = None,
) -> int:
    if start_pos is None:
        start_pos = (0.0, 0.0, 0.48)
    if start_rpy is None:
        start_rpy = (0.0, 0.0, 0.0)
    quat = p.getQuaternionFromEuler(start_rpy)
    return p.loadURDF(LAIKAGO_URDF, start_pos, quat, useFixedBase=False)


def load_minitaur(
    start_pos: Sequence[float] | None = None,
) -> int:
    if start_pos is None:
        start_pos = (0.0, 0.0, 0.2)
    return p.loadURDF(
        MINITAUR_URDF,
        start_pos,
        p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
    )


def minitaur_motor_indices(robot: int) -> list[int]:
    m: dict[str, int] = {}
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        m[info[1].decode("utf-8")] = i
    return [m[n] for n in MINITAUR_MOTOR_NAMES]


def apply_laikago_sdk(
    robot: int,
    joint_indices: Sequence[int],
    sdk: np.ndarray | Sequence[float],
    max_force: float = 55.0,
) -> None:
    sdk = np.asarray(sdk, dtype=np.float64)
    urdf = sdk_to_urdf(sdk)
    for ji, pos in zip(joint_indices, urdf):
        p.setJointMotorControl2(
            robot, int(ji), p.POSITION_CONTROL, targetPosition=float(pos), force=max_force
        )


def reset_laikago_pose(robot: int, joint_indices: Sequence[int], sdk: np.ndarray | None = None) -> None:
    if sdk is None:
        sdk = np.array(INIT_SDK, dtype=np.float64)
    else:
        sdk = np.asarray(sdk, dtype=np.float64)
    urdf = sdk_to_urdf(sdk)
    for ji, pos in zip(joint_indices, urdf):
        p.resetJointState(robot, int(ji), float(pos))


def trot_sdk_targets(t: float, freq_hz: float, init_sdk: np.ndarray | None = None) -> np.ndarray:
    if init_sdk is None:
        base = np.array(INIT_SDK, dtype=np.float64)
    else:
        base = np.array(init_sdk, dtype=np.float64)
    w = 2.0 * math.pi * freq_hz
    phase = (math.pi, 0.0, 0.0, math.pi)
    out = base.copy()
    for leg in range(4):
        s = math.sin(w * t + phase[leg])
        i = leg * 3
        out[i + 0] += 0.08 * s
        out[i + 1] += 0.22 * s
        out[i + 2] += -0.28 * s
    return out


def foot_on_ground(robot: int, toe_link: int, threshold: float = 0.02) -> bool:
    pts = p.getContactPoints(bodyA=robot, linkIndexA=toe_link)
    return len(pts) > 0


def any_laikago_contact(robot: int) -> bool:
    pts = p.getContactPoints(bodyA=robot)
    return len(pts) > 0
