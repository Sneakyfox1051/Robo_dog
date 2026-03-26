"""Laikago spawn + gravity drop check (headless)."""

from __future__ import annotations

import pybullet as p

import robo_common as rc


def main() -> None:
    cid = rc.connect(gui=False)
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        rc.reset_laikago_pose(robot, j)
        z0 = p.getBasePositionAndOrientation(robot)[0][2]
        for _ in range(600):
            rc.apply_laikago_sdk(robot, j, rc.INIT_SDK)
            p.stepSimulation()
        z1 = p.getBasePositionAndOrientation(robot)[0][2]
        print(f"test3.py: OK  height {z0:.3f} -> {z1:.3f}")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
