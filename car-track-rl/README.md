# Car Track RL

This project mirrors the structure of `flappy-bird-rl`, but trains a PPO agent to drive a simple top-down car around a closed track.

## Layout

- `train_ppo.py`: PPO training
- `evaluate.py`: load a checkpoint and watch or benchmark the policy
- `play_manual.py`: drive the car yourself with the keyboard
- `car_rl/`: environment, policy, and wrapper code
- `artifacts/`: default checkpoint output directory

## Environment

The environment is a lightweight Gymnasium-style top-down track:

- discrete actions: coast, accelerate, brake, left, right, accelerate-left, accelerate-right
- vector observations: speed, lateral offset, heading error, recent progress, and five ray distances
- reward: forward progress with penalties for drifting off-center, over-steering, idling, and leaving the track

The default renderer uses `pygame`.

## Setup

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Train

```bash
./.venv/bin/python train_ppo.py --device auto
```

Example with a larger model and frame stacking:

```bash
./.venv/bin/python train_ppo.py --hidden-dims 256,256 --frame-stack 2
```

## Evaluate

Run a deterministic evaluation without rendering:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --episodes 20
```

Watch the learned policy:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --render human --delay 0.02
```

## Manual Play

Use arrow keys or `WASD`:

```bash
./.venv/bin/python play_manual.py
```
