"""Minitaur motor sliders (GUI)."""

from __future__ import annotations

import math

import pybullet as p

import robo_common as rc


def main() -> None:
    cid = rc.connect(gui=True)
    try:
        rc.setup_scene()
        rid = rc.load_minitaur()
        mids = rc.minitaur_motor_indices(rid)
        sliders: list[int] = []
        for name, mid in zip(rc.MINITAUR_MOTOR_NAMES, mids):
            sliders.append(
                p.addUserDebugParameter(name, -math.pi / 2, math.pi / 2, 0.0)
            )

        p.setRealTimeSimulation(1)
        while p.isConnected(cid):
            for mid, s in zip(mids, sliders):
                ang = p.readUserDebugParameter(s)
                p.setJointMotorControl2(
                    rid, mid, p.POSITION_CONTROL, targetPosition=ang, force=15.0
                )
    finally:
        if p.isConnected(cid):
            p.disconnect(cid)


if __name__ == "__main__":
    main()
