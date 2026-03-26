# Robo_dog

PyBullet quadruped (Laikago/Minitaur) simulation + reinforcement learning experiments (PPO).

## Setup

```powershell
cd C:\robo_delete
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick demos (GUI)

```powershell
python gui_walk_demo.py
python auto_walk.py
python walking_sine_waves.py
python slider.py
```

## Train + showcase stable walk

```powershell
python train_stable_walk.py --timesteps 600000 --save stable_walk --device cpu
python show_stable_walk.py --model stable_walk --episodes 3
```

