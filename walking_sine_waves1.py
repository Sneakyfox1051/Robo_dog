"""Sine-wave walk with per-leg frequency and amplitude scaling (GUI)."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pybullet as p

import robo_common as rc
import sine_waves_base as sw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--freq", type=float, default=1.25)
    ap.add_argument("--amp", type=float, default=1.15, help="Scale leg motion amplitude.")
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
        phases = (0.0, math.pi, math.pi * 0.95, 0.05)
        while True:
            now = time.time() - t0
            if now > args.duration:
                break
            base = np.array(rc.INIT_SDK, dtype=np.float64)
            for leg in range(4):
                off = sw.sine_leg_offsets(now, args.freq, phases[leg]) * args.amp
                base[leg * 3 : leg * 3 + 3] += off
            rc.apply_laikago_sdk(robot, j, base)
            if args.direct:
                p.stepSimulation()
            else:
                time.sleep(dt)
        print("walking_sine_waves1.py: done")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
