"""Short environment check: Laikago + contact query."""

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
        for _ in range(400):
            rc.apply_laikago_sdk(robot, j, rc.INIT_SDK)
            p.stepSimulation()
        n = len(p.getContactPoints(bodyA=robot))
        print(f"test4.py: OK  contacts={n}")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
