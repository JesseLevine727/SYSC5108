# Car Track RL

This project mirrors the structure of `flappy-bird-rl`, but trains a PPO agent to drive a simple top-down car around a closed track.

## Layout

- `train_ppo.py`: PPO training
- `evaluate.py`: load a checkpoint and watch or benchmark the policy
- `benchmark_generalization.py`: benchmark a checkpoint across multiple randomization regimes
- `play_manual.py`: drive the car yourself with the keyboard
- `car_rl/`: environment, policy, and wrapper code
- `artifacts/`: default checkpoint output directory

## Environment

The environment is a lightweight Gymnasium-style top-down track with domain randomization:

- discrete actions: coast, accelerate, brake, left, right, accelerate-left, accelerate-right
- vector observations: speed, lateral offset, heading error, recent progress, and five ray distances
- multiple procedural track generators: radial, ellipse, peanut, and stadium-like families
- randomized width, scale, harmonic deformation, start pose, vehicle dynamics, and sensor noise
- randomized driving dynamics and observation noise during training and evaluation
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

The trainer ramps domain randomization up over time and stops early when the policy is consistently solved on held-out random tracks.

## Evaluate

Run a deterministic evaluation without rendering:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --episodes 20
```

Watch the learned policy:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --render human --delay 0.02
```

Override the evaluation randomization level:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --domain-randomization-scale 1.25
```

## Generalization Benchmark

Benchmark a checkpoint on baseline, randomized, and stress-randomized settings:

```bash
./.venv/bin/python benchmark_generalization.py --checkpoint artifacts/ppo_solved_model.pt --episodes 24
```

## Manual Play

Use arrow keys or `WASD`:

```bash
./.venv/bin/python play_manual.py
```
