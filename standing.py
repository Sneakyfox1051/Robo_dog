"""Hold nominal standing pose (GUI / optional headless)."""

from __future__ import annotations

import argparse
import time

import pybullet as p

import robo_common as rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    cid = rc.connect(gui=not args.direct)
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        rc.reset_laikago_pose(robot, j)
        p.setRealTimeSimulation(1 if not args.direct else 0)
        dt = 1.0 / 240.0
        p.setTimeStep(dt)
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            rc.apply_laikago_sdk(robot, j, rc.INIT_SDK)
            if args.direct:
                p.stepSimulation()
            else:
                time.sleep(dt)
        z = p.getBasePositionAndOrientation(robot)[0][2]
        print(f"standing.py: OK  z={z:.3f}")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
