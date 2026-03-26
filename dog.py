"""Minitaur: simple periodic motor sweep (GUI)."""

from __future__ import annotations

import argparse
import math
import time

import pybullet as p

import robo_common as rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--duration", type=float, default=15.0)
    args = ap.parse_args()

    cid = rc.connect(gui=not args.direct)
    try:
        rc.setup_scene()
        rid = rc.load_minitaur()
        mids = rc.minitaur_motor_indices(rid)
        p.setRealTimeSimulation(1 if not args.direct else 0)
        dt = 1.0 / 240.0
        p.setTimeStep(dt)
        t0 = time.time()
        while time.time() - t0 < args.duration:
            t = time.time() - t0
            for k, mid in enumerate(mids):
                ang = 0.4 * math.sin(2 * math.pi * 0.7 * t + k * 0.5)
                p.setJointMotorControl2(
                    rid, mid, p.POSITION_CONTROL, targetPosition=ang, force=8.0
                )
            if args.direct:
                p.stepSimulation()
            else:
                time.sleep(dt)
        print("dog.py: done")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
