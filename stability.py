"""Simple balance heuristic: reduce roll/pitch via hip abduction offsets (GUI)."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p

import robo_common as rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--gain", type=float, default=0.35)
    args = ap.parse_args()

    cid = rc.connect(gui=not args.direct)
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        sdk = np.array(rc.INIT_SDK, dtype=np.float64)
        rc.reset_laikago_pose(robot, j, sdk)
        p.setRealTimeSimulation(1 if not args.direct else 0)
        dt = 1.0 / 240.0
        p.setTimeStep(dt)
        t0 = time.time()
        while time.time() - t0 < args.duration:
            _, quat = p.getBasePositionAndOrientation(robot)
            roll, pitch, _ = p.getEulerFromQuaternion(quat)
            corr = sdk.copy()
            corr[0] -= args.gain * roll
            corr[3] -= args.gain * roll
            corr[6] -= args.gain * roll
            corr[9] -= args.gain * roll
            corr[1] -= 0.15 * pitch
            corr[4] -= 0.15 * pitch
            corr[7] -= 0.15 * pitch
            corr[10] -= 0.15 * pitch
            rc.apply_laikago_sdk(robot, j, corr)
            if args.direct:
                p.stepSimulation()
            else:
                time.sleep(dt)
        print("stability.py: done")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
