"""Scripted Laikago trot (same core as gui_walk demo, runnable script)."""

from __future__ import annotations

import argparse
import time

import pybullet as p

import robo_common as rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--freq", type=float, default=1.25)
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
        while time.time() - t0 < args.duration:
            now = time.time() - t0
            sdk = rc.trot_sdk_targets(now, args.freq)
            rc.apply_laikago_sdk(robot, j, sdk)
            if args.direct:
                p.stepSimulation()
            else:
                time.sleep(dt)
        print("auto_walk.py: done")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
