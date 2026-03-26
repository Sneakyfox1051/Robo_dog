"""Forward-motion trial: bias hip joints + light trot (GUI)."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p

import robo_common as rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--duration", type=float, default=18.0)
    args = ap.parse_args()

    cid = rc.connect(gui=not args.direct)
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        base = np.array(rc.INIT_SDK, dtype=np.float64)
        base[1] += 0.08
        base[4] += 0.08
        base[7] += 0.08
        base[10] += 0.08
        rc.reset_laikago_pose(robot, j, base)
        p.setRealTimeSimulation(1 if not args.direct else 0)
        dt = 1.0 / 240.0
        p.setTimeStep(dt)
        t0 = time.time()
        while time.time() - t0 < args.duration:
            now = time.time() - t0
            sdk = rc.trot_sdk_targets(now, 1.15, init_sdk=base)
            rc.apply_laikago_sdk(robot, j, sdk)
            if args.direct:
                p.stepSimulation()
            else:
                time.sleep(dt)
        lin, _ = p.getBaseVelocity(robot)
        print(f"forward.py: done  vx={lin[0]:.3f}")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
