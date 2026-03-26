"""Short loop: FR foot IK targets stepping forward with tiny RL-style noise."""

from __future__ import annotations

import argparse
import time

import numpy as np
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
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        rc.reset_laikago_pose(robot, j)
        toe = rc.link_index_from_name(robot, "toeFR")
        fr = j[0:3]
        rng = np.random.default_rng(0)
        p.setRealTimeSimulation(1 if not args.direct else 0)
        dt = 1.0 / 240.0
        p.setTimeStep(dt)
        t0 = time.time()
        phase = 0.0
        while time.time() - t0 < args.duration:
            pos = p.getLinkState(robot, toe)[0]
            phase += 0.04
            target = [
                pos[0] + 0.04 + 0.01 * np.sin(phase),
                pos[1] + float(rng.normal(0, 0.005)),
                0.06 + 0.01 * float(rng.normal(0, 1)),
            ]
            try:
                sol = p.calculateInverseKinematics(
                    robot,
                    toe,
                    target,
                    lowerLimits=[-3.0] * 3,
                    upperLimits=[8.0] * 3,
                    jointRanges=[6.0] * 3,
                    restPoses=[p.getJointState(robot, x)[0] for x in fr],
                    maxNumIterations=35,
                    jointIndices=fr,
                )
                blend = 0.25
                for idx, ang in zip(fr, sol):
                    cur = p.getJointState(robot, idx)[0]
                    p.resetJointState(robot, idx, (1 - blend) * cur + blend * float(ang))
            except Exception:
                pass
            sdk = rc.trot_sdk_targets(time.time() - t0, 1.0)
            rc.apply_laikago_sdk(robot, j, sdk)
            if args.direct:
                p.stepSimulation()
            else:
                time.sleep(dt)
        print("ik_walk.py: done")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
