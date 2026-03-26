"""Laikago open-loop trot in PyBullet — good for checking GUI + sim.



Run (with venv active):

  python gui_walk_demo.py              # GUI window

  python gui_walk_demo.py --direct     # no window, fast sanity check



Close the GUI window or press Ctrl+C in the terminal to stop.

"""



from __future__ import annotations



import argparse

import time



import pybullet as p



import robo_common as rc





def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument("--direct", action="store_true", help="Run without GUI (smoke test).")

    ap.add_argument("--freq", type=float, default=1.25, help="Stride frequency (Hz).")

    ap.add_argument(

        "--duration",

        type=float,

        default=30.0,

        help="Seconds to run (set 0 to run until Ctrl+C).",

    )

    args = ap.parse_args()



    cid = rc.connect(gui=not args.direct)

    if not args.direct:

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    try:

        rc.setup_scene()

        robot = rc.load_laikago()

        indices = rc.laikago_joint_indices(robot)

        rc.reset_laikago_pose(robot, indices)



        p.setRealTimeSimulation(1 if not args.direct else 0)

        timestep = 1.0 / 240.0

        p.setTimeStep(timestep)

        max_force = 55.0

        t0 = time.time()

        print("Running Laikago trot demo. Close GUI or Ctrl+C to stop.")



        while True:

            now = time.time() - t0

            if args.duration > 0 and now >= args.duration:

                break

            sdk = rc.trot_sdk_targets(now, args.freq)

            rc.apply_laikago_sdk(robot, indices, sdk, max_force=max_force)

            if args.direct:

                p.stepSimulation()

            else:

                time.sleep(timestep)

    finally:

        p.disconnect(cid)





if __name__ == "__main__":

    main()


