# Flappy Bird RL

This workspace now contains a pygame-based Flappy Bird reinforcement learning project that trains with PyTorch and uses the GPU automatically when CUDA is available.

## Environment

- Bird actions: `0 = glide`, `1 = flap`
- Observation: bird height, vertical velocity, distance to the next pipe, and vertical offset to the next gap
- Reward shaping: small survival reward, bonus for passing a pipe, large penalty for crashing

The game is rendered with `pygame` and uses Flappy Bird-style screen size, scrolling pipes, gravity, and floor collisions. Training stays headless by default for speed, but the physics are the same ones used for human rendering.

## Setup

Create and use the local virtual environment:

```bash
./.venv/bin/pip install -r requirements.txt
```

PyTorch in this `.venv` was installed with CUDA support and should use your NVIDIA GPU by default.

## Train

```bash
./.venv/bin/python train.py --episodes 800
```

Reliable passing baseline:

```bash
./.venv/bin/python train_clone.py --device cuda
```

Strict RL with PPO:

```bash
./.venv/bin/python train_ppo.py --device cuda
```

`train.py` is the experimental DQN trainer. `train_clone.py` is the practical path when you want a model that visibly passes pipes right now.
`train_ppo.py` is the pure-RL trainer with no expert actions or cloning. It saves:

- `artifacts/ppo_best_model.pt`
- `artifacts/ppo_last_model.pt`

Useful flags:

- `--device cuda` forces GPU usage and fails fast if CUDA is not available
- `--episodes 2000` trains longer for a stronger policy
- `--checkpoint-dir artifacts` controls where checkpoints are saved

The trainer writes:

- `artifacts/best_model.pt`
- `artifacts/last_model.pt`

## Evaluate

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/best_model.pt --episodes 10
```

Pygame playback:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/best_model.pt --render human
```

Strict-RL PPO playback:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --render human --device cuda --max-steps 20000
```

Text-mode playback:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/best_model.pt --render text
```

Manual play:

```bash
./.venv/bin/python play_manual.py
```

## CUDA sanity check

```bash
./.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
```
