# Flappy Bird RL

This is the cleaned project directory for the Flappy Bird work. It contains only the active PPO codepath, the pygame environment, the manual play script, and the main PPO checkpoints.

## Layout

- `train_ppo.py`: PPO training and fine-tuning
- `evaluate.py`: PPO evaluation and playback
- `play_manual.py`: play the environment yourself
- `flappy_rl/`: environment and PPO support code
- `artifacts/ppo_best_model.pt`: current 4-state production baseline
- `artifacts/ppo_5state_best_model.pt`: tuned 5-state PPO with `predicted_gap_error_at_crossing`

## Setup

Install dependencies in a virtual environment:

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

If you already have a working environment elsewhere, you can use that Python instead.

## Current Model

The current baseline model is:

- PPO
- actor-critic MLP
- base observation: bird height, velocity, next-pipe distance, next-gap offset

The trainer can also optionally add:

- `predicted_gap_error_at_crossing`

via `--use-predicted-gap-error`.

The promoted 5-state checkpoint is stored in:

- `artifacts/ppo_5state_best_model.pt`

## Train

Fine-tune from the current production checkpoint:

```bash
./.venv/bin/python train_ppo.py --device cuda
```

Train with the predicted crossing-error feature:

```bash
./.venv/bin/python train_ppo.py --device cuda --use-predicted-gap-error --checkpoint-dir artifacts_predgap
```

## Evaluate

Watch the current 4-state baseline:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --device cuda --render human --max-steps 100000
```

Evaluate without rendering:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --device cuda --episodes 30 --max-steps 20000
```

Watch the tuned 5-state model:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_5state_best_model.pt --device cuda --render human --max-steps 100000
```

## Manual Play

```bash
./.venv/bin/python play_manual.py
```
