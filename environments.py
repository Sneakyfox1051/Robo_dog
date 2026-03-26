"""Gymnasium environments for Laikago in PyBullet (balance + trot walking)."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

import robo_common as rc


class LaikagoBulletEnv(gym.Env):
    """Base: 12-dim actions as scaled deltas in SDK joint space."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 800,
        action_scale: float = 0.12,
        gui: bool = False,
        action_smoothing: float = 0.85,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.gui = gui or (render_mode == "human")
        self.action_smoothing = float(action_smoothing)

        self._client: Optional[int] = None
        self.robot: int = -1
        self.joint_indices: list[int] = []
        self._sdk = np.array(rc.INIT_SDK, dtype=np.float32)
        self._step_no = 0
        self._a_prev = np.zeros((12,), dtype=np.float32)
        self._a_filt = np.zeros((12,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
        )

    def _connect_if_needed(self) -> None:
        if self._client is not None:
            return
        self._client = rc.connect(gui=self.gui)
        rc.setup_scene(plane=True)

    def _reload_robot(self) -> None:
        p.resetSimulation()
        rc.setup_scene(plane=True)
        self.robot = rc.load_laikago()
        self.joint_indices = rc.laikago_joint_indices(self.robot)
        rc.reset_laikago_pose(self.robot, self.joint_indices, self._sdk)

    def _get_obs(self) -> np.ndarray:
        pos, quat = p.getBasePositionAndOrientation(self.robot)
        lin, ang = p.getBaseVelocity(self.robot)
        jpos: list[float] = []
        jvel: list[float] = []
        for ji in self.joint_indices:
            st = p.getJointState(self.robot, ji)
            jpos.append(st[0])
            jvel.append(st[1])
        return np.concatenate(
            [pos, quat, ang, lin, np.array(jpos, dtype=np.float32), np.array(jvel, dtype=np.float32)],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        self._step_no = 0
        # Small reset noise helps policies become robust.
        rng = np.random.default_rng(seed)
        self._sdk = (np.array(rc.INIT_SDK, dtype=np.float32) + rng.normal(0, 0.03, size=(12,)).astype(np.float32))
        self._connect_if_needed()
        self._reload_robot()
        self._a_prev[:] = 0.0
        self._a_filt[:] = 0.0
        for _ in range(10):
            rc.apply_laikago_sdk(self.robot, self.joint_indices, self._sdk)
            p.stepSimulation()
        return self._get_obs(), {}

    def _filter_action(self, a: np.ndarray) -> np.ndarray:
        # Low-pass filter to discourage jerky control.
        alpha = np.clip(self.action_smoothing, 0.0, 0.98)
        self._a_filt = alpha * self._a_filt + (1.0 - alpha) * a
        return self._a_filt

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def render(self):
        if self.render_mode == "rgb_array":
            w, h, rgba, _, _ = p.getCameraImage(320, 200)
            return rgba[:, :, :3]
        return None

    def close(self) -> None:
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None


class DogBalanceEnv(LaikagoBulletEnv):
    """Stay upright with reasonable height."""

    def step(self, action: np.ndarray):
        self._step_no += 1
        a_raw = np.clip(action.astype(np.float32), -1.0, 1.0)
        a = self._filter_action(a_raw)
        self._sdk = np.clip(self._sdk + a * self.action_scale, -2.5, 2.5)
        rc.apply_laikago_sdk(self.robot, self.joint_indices, self._sdk)
        p.stepSimulation()

        pos, quat = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        z = pos[2]
        upright = float(np.exp(-4.0 * (roll * roll + pitch * pitch)))
        height_term = np.clip((z - 0.32) * 2.0, -1.0, 1.0)
        smooth_pen = float(np.dot(a_raw - self._a_prev, a_raw - self._a_prev))
        self._a_prev = a_raw
        reward = 1.0 * upright + 0.4 * height_term
        reward -= 0.004 * float(np.dot(a_raw, a_raw))
        reward -= 0.002 * smooth_pen

        terminated = z < 0.22 or abs(roll) > 1.15 or abs(pitch) > 1.15
        truncated = self._step_no >= self.max_episode_steps
        return self._get_obs(), reward, terminated, truncated, {}


class TrotWalkEnv(LaikagoBulletEnv):
    """Reward forward +x velocity while staying somewhat upright."""

    def step(self, action: np.ndarray):
        self._step_no += 1
        a_raw = np.clip(action.astype(np.float32), -1.0, 1.0)
        a = self._filter_action(a_raw)
        self._sdk = np.clip(self._sdk + a * self.action_scale, -2.5, 2.5)
        rc.apply_laikago_sdk(self.robot, self.joint_indices, self._sdk)
        p.stepSimulation()

        pos, quat = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        lin, ang = p.getBaseVelocity(self.robot)
        z = pos[2]
        vx = lin[0]
        vy = lin[1]
        wz = ang[2]

        # Shaped rewards: forward progress + alive/upright + stable height,
        # plus penalties to reduce spinning and sideways drift.
        upright = float(np.exp(-5.0 * (roll * roll + pitch * pitch)))
        height_ok = float(np.exp(-25.0 * (z - 0.37) * (z - 0.37)))
        reward = 1.6 * float(np.clip(vx, -1.0, 2.0))
        reward += 0.8 * upright
        reward += 0.35 * height_ok
        reward -= 0.25 * float(abs(vy))
        reward -= 0.05 * float(abs(wz))

        smooth_pen = float(np.dot(a_raw - self._a_prev, a_raw - self._a_prev))
        self._a_prev = a_raw
        reward -= 0.006 * float(np.dot(a_raw, a_raw))
        reward -= 0.003 * smooth_pen

        terminated = z < 0.22 or abs(roll) > 1.1 or abs(pitch) > 1.1
        truncated = self._step_no >= self.max_episode_steps
        return self._get_obs(), float(reward), terminated, truncated, {}


class IKAugmentedWalkEnv(TrotWalkEnv):
    """Same as walk env, but gently pulls FR foot toward an IK target each step."""

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._toe_fr = rc.link_index_from_name(self.robot, "toeFR")
        self._fr_leg_ids = self.joint_indices[0:3]
        return obs, info

    def step(self, action: np.ndarray):
        obs, rew, term, trunc, info = super().step(action)
        toe_pos = p.getLinkState(self.robot, self._toe_fr)[0]
        target = [
            toe_pos[0] + 0.02,
            toe_pos[1],
            max(0.05, toe_pos[2] - 0.01),
        ]
        try:
            sol = p.calculateInverseKinematics(
                self.robot,
                self._toe_fr,
                target,
                lowerLimits=[-3.0] * 3,
                upperLimits=[8.0] * 3,
                jointRanges=[5.0] * 3,
                restPoses=[p.getJointState(self.robot, j)[0] for j in self._fr_leg_ids],
                maxNumIterations=40,
                residualThreshold=1e-4,
                jointIndices=self._fr_leg_ids,
            )
            blend = 0.08
            for k, j in enumerate(self._fr_leg_ids):
                cur = p.getJointState(self.robot, j)[0]
                p.resetJointState(
                    self.robot, j, float((1 - blend) * cur + blend * sol[k])
                )
            p.stepSimulation()
        except Exception:
            pass
        obs = self._get_obs()
        return obs, rew, term, trunc, info
