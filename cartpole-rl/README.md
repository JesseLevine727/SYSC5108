# CartPole RL

This directory contains a standalone PPO trainer for the CartPole control problem. It mirrors the `flappy-bird-rl` project layout, but uses a local CartPole environment so it does not depend on `gymnasium`.

## Layout

- `train_ppo.py`: PPO training and checkpointing
- `evaluate.py`: checkpoint evaluation and rollout playback
- `train_dqn.py`: DQN training for more stable solved CartPole performance
- `evaluate_dqn.py`: DQN checkpoint evaluation
- `cartpole_rl/`: CartPole environment and policy network
- `artifacts/`: saved checkpoints

## Setup

Install dependencies in a virtual environment:

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

If you are running from the shared workspace root, use `./.venv/bin/python cartpole-rl/...`. If you `cd cartpole-rl`, use `../.venv/bin/python ...`.

## Train

Train a CartPole agent:

```bash
../.venv/bin/python train_ppo.py --device auto
```

A short smoke-test run:

```bash
../.venv/bin/python train_ppo.py --updates 4 --num-envs 8 --rollout-steps 128 --eval-every 2 --eval-episodes 8
```

## Evaluate

Evaluate the best saved checkpoint:

```bash
../.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --episodes 10
```

Render a text rollout in the terminal:

```bash
../.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --episodes 3 --render text --delay 0.03
```

## DQN

Train a DQN agent:

```bash
../.venv/bin/python train_dqn.py --device cuda
```

Evaluate the best DQN checkpoint:

```bash
../.venv/bin/python evaluate_dqn.py --checkpoint dqn_artifacts/dqn_best_model.pt --episodes 10 --device cuda
```

Open a simple pygame demo window for the solved DQN checkpoint:

```bash
../.venv/bin/python evaluate_dqn.py --checkpoint dqn_gpu_tuned/dqn_best_model.pt --episodes 3 --device cuda --render human --delay 0.03
```
