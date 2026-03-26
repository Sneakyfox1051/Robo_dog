"""Basic validation: base linear velocity after small forward tilt."""

from __future__ import annotations

import numpy as np
import pybullet as p

import robo_common as rc


def main() -> None:
    cid = rc.connect(gui=False)
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        sdk = np.array(rc.INIT_SDK, dtype=np.float64)
        sdk[1] += 0.05
        sdk[4] += 0.05
        sdk[7] += 0.05
        sdk[10] += 0.05
        rc.reset_laikago_pose(robot, j, sdk)
        for _ in range(800):
            rc.apply_laikago_sdk(robot, j, sdk)
            p.stepSimulation()
        lin, _ = p.getBaseVelocity(robot)
        print(f"test5.py: OK  vx={lin[0]:.3f} vy={lin[1]:.3f}")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
