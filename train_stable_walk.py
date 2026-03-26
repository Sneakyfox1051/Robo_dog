"""Train a stable forward-walking policy for Laikago (PyBullet) using PPO.

This writes:
- stable_walk.zip (policy)
- stable_walk_vecnorm.pkl (obs/reward normalization stats)

Run:
  .\venv\Scripts\Activate.ps1
  python train_stable_walk.py --timesteps 600000
"""

from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments import TrotWalkEnv


def make_env():
    return Monitor(TrotWalkEnv(gui=False, max_episode_steps=1200, action_scale=0.10, action_smoothing=0.88))


def make_eval_env():
    return Monitor(TrotWalkEnv(gui=False, max_episode_steps=1200, action_scale=0.10, action_smoothing=0.88))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--save", type=str, default="stable_walk")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--tb", type=str, default="tb_runs", help="TensorBoard log dir")
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    args = ap.parse_args()

    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Keep eval normalization synced with training.
    eval_env.obs_rms = venv.obs_rms

    ckpt = CheckpointCallback(save_freq=100_000, save_path="checkpoints", name_prefix=args.save)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="best_model",
        log_path="eval_logs",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        device=args.device,
        tensorboard_log=args.tb,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=[ckpt, eval_cb])
    print(f"train time: {time.time() - t0:.1f}s")

    model.save(args.save)
    venv.save(args.save + "_vecnorm.pkl")
    venv.close()
    eval_env.close()
    print(f"train_stable_walk.py: saved {args.save}.zip and {args.save}_vecnorm.pkl")


if __name__ == "__main__":
    main()

