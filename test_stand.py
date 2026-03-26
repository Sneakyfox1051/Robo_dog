"""Evaluate stand model (laikago_stand_model) or fallback to balance_model."""

from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from environments import DogBalanceEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    for name in ("laikago_stand_model", "balance_model"):
        if os.path.isfile(name + ".zip"):
            mp = name
            break
    else:
        raise SystemExit(
            "No laikago_stand_model.zip or balance_model.zip — run RL_stand.py or stage1_rl.py."
        )

    print("Using", mp + ".zip")
    env = DogBalanceEnv(gui=args.gui)
    model = PPO.load(mp, env=env)
    for ep in range(3):
        obs, _ = env.reset()
        g = 0.0
        for _ in range(600):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(np.array(a))
            g += float(r)
            if term or trunc:
                break
        print(f"test_stand episode {ep + 1}: return={g:.2f}")
    env.close()


if __name__ == "__main__":
    main()
