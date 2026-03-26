"""Single inverse-kinematics solve for the front-right toe (GUI)."""

from __future__ import annotations

import argparse
import time

import pybullet as p

import robo_common as rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true")
    args = ap.parse_args()

    cid = rc.connect(gui=not args.direct)
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        rc.reset_laikago_pose(robot, j)
        toe = rc.link_index_from_name(robot, "toeFR")
        fr = j[0:3]
        pos0 = p.getLinkState(robot, toe)[0]
        target = [pos0[0] + 0.05, pos0[1], max(0.04, pos0[2] - 0.03)]
        sol = p.calculateInverseKinematics(
            robot,
            toe,
            target,
            lowerLimits=[-3.0] * 3,
            upperLimits=[8.0] * 3,
            jointRanges=[6.0] * 3,
            restPoses=[p.getJointState(robot, x)[0] for x in fr],
            maxNumIterations=60,
            jointIndices=fr,
        )
        for idx, ang in zip(fr, sol):
            p.resetJointState(robot, idx, float(ang))
        print("ik.py: IK target", target, "applied to FR leg")

        p.setRealTimeSimulation(1 if not args.direct else 0)
        for _ in range(360):
            rc.apply_laikago_sdk(robot, j, rc.INIT_SDK)
            if args.direct:
                p.stepSimulation()
            else:
                time.sleep(1.0 / 240.0)
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
