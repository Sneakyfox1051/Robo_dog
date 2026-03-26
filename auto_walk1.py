"""Trot with periodic foot contact reporting (terminal dashboard)."""

from __future__ import annotations

import argparse
import time

import pybullet as p

import robo_common as rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--freq", type=float, default=1.2)
    args = ap.parse_args()

    cid = rc.connect(gui=not args.direct)
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        rc.reset_laikago_pose(robot, j)
        toe_idx = [rc.link_index_from_name(robot, n) for n in rc.LAIKAGO_TOE_LINK_NAMES]
        p.setRealTimeSimulation(1 if not args.direct else 0)
        dt = 1.0 / 240.0
        p.setTimeStep(dt)
        t0 = time.time()
        last_print = 0.0
        while time.time() - t0 < args.duration:
            now = time.time() - t0
            sdk = rc.trot_sdk_targets(now, args.freq)
            rc.apply_laikago_sdk(robot, j, sdk)
            if args.direct:
                p.stepSimulation()
            else:
                time.sleep(dt)
            if now - last_print > 0.5:
                last_print = now
                feet = [rc.foot_on_ground(robot, ti) for ti in toe_idx]
                print(f"t={now:5.2f}s  feet_FR_FL_RR_RL={feet}")
        print("auto_walk1.py: done")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
