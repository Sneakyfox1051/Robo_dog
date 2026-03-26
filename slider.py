"""Debug-parameter sliders for Laikago SDK joint targets (GUI)."""

from __future__ import annotations

import numpy as np
import pybullet as p

import robo_common as rc


def main() -> None:
    cid = rc.connect(gui=True)
    try:
        rc.setup_scene()
        robot = rc.load_laikago()
        j = rc.laikago_joint_indices(robot)
        sdk_nom = np.array(rc.INIT_SDK, dtype=np.float64)
        sliders: list[int] = []
        part = ("abd", "hip", "knee")
        for leg in ("FR", "FL", "RR", "RL"):
            for k in range(3):
                sliders.append(
                    p.addUserDebugParameter(
                        f"{leg}_{part[k]}", -0.85, 0.85, 0.0
                    )
                )

        p.setRealTimeSimulation(1)
        while p.isConnected(cid):
            sdk = sdk_nom + np.array(
                [p.readUserDebugParameter(s) for s in sliders], dtype=np.float64
            )
            rc.apply_laikago_sdk(robot, j, sdk)
    finally:
        if p.isConnected(cid):
            p.disconnect(cid)


if __name__ == "__main__":
    main()
