"""Create actions.npy sample (joint deltas history) for replay / debugging."""

from __future__ import annotations

import numpy as np

N = 1000
actions = (np.random.randn(N, 12) * 0.15).astype(np.float32)
np.save("actions.npy", actions)
print("make_actions.py: wrote actions.npy shape", actions.shape)
