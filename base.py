"""Minimal baseline: plane + Laikago, a few physics steps."""

from __future__ import annotations

import pybullet as p

import robo_common as rc


def main() -> None:
    cid = rc.connect(gui=True)
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        joints = rc.laikago_joint_indices(robot)
        rc.reset_laikago_pose(robot, joints)
        for _ in range(240):
            rc.apply_laikago_sdk(robot, joints, rc.INIT_SDK)
            p.stepSimulation()
        print("base.py: OK — Laikago loaded and stepped.")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
