"""Quick PyBullet sanity check (no robot)."""

from __future__ import annotations

import pybullet as p
import pybullet_data


def main() -> None:
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        for _ in range(120):
            p.stepSimulation()
        print("test2.py: OK")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
