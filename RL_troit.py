"""Open-loop trot with observation vectors printed (simple 'dashboard')."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p

import robo_common as rc


def collect_obs(robot: int, joint_indices: list[int]) -> np.ndarray:
    pos, quat = p.getBasePositionAndOrientation(robot)
    lin, ang = p.getBaseVelocity(robot)
    jp = [p.getJointState(robot, ji)[0] for ji in joint_indices]
    jv = [p.getJointState(robot, ji)[1] for ji in joint_indices]
    return np.concatenate([pos, quat, ang, lin, jp, jv]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--duration", type=float, default=12.0)
    args = ap.parse_args()

    cid = rc.connect(gui=not args.direct)
    buf: list[np.ndarray] = []
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        rc.reset_laikago_pose(robot, j)
        t0 = time.time()
        while time.time() - t0 < args.duration:
            now = time.time() - t0
            sdk = rc.trot_sdk_targets(now, 1.2)
            rc.apply_laikago_sdk(robot, j, sdk)
            p.stepSimulation()
            buf.append(collect_obs(robot, j))
        if buf:
            stacked = np.stack(buf, axis=0)
            print("RL_troit.py: obs dim", stacked.shape[1])
            print(
                "  mean base z", float(np.mean(stacked[:, 2])),
                "std vz", float(np.std(stacked[:, 12])),
            )
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
