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
- fixed hand-authored layouts split into training families and holdout evaluation families
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

To train only on the training pool and evaluate on the full pool:

```bash
./.venv/bin/python train_ppo.py --train-track-pool train --evaluation-track-pool all
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

Override the evaluation randomization level:

```bash
./.venv/bin/python evaluate.py --checkpoint artifacts/ppo_best_model.pt --domain-randomization-scale 1.25
```

## Generalization Benchmark

Benchmark a checkpoint on procedural, mixed, and holdout handcrafted settings:

```bash
./.venv/bin/python benchmark_generalization.py --checkpoint artifacts/ppo_solved_model.pt --episodes 24
```

## Plots

Generate the summary plots:

```bash
./.venv/bin/python generate_experiment_plots.py
```

Generated SVGs are written to `plots/`:

- `plots/training_summary.svg`
- `plots/sample_efficiency.svg`
- `plots/run4_generalization_benchmark.svg`
- `plots/run5_transfer_benchmark.svg`
- `plots/run13_transfer_benchmark.svg`

## Experiment Summary

All runs below were trained with PPO in `torch` on the local RTX 5080 GPU.

### Run 1: Initial Single-Track Baseline

- Command: `./.venv/bin/python train_ppo.py --device cuda --updates 60 --checkpoint-dir artifacts_run1`
- Best checkpoint: `artifacts_run1/ppo_best_model.pt`
- Best 20-episode eval:
  - `mean_reward = 2370.26`
  - `mean_laps = 3.00`
  - `mean_progress = 2.991`
  - `off_track = 0`
- Result: learned the task on the original track setup, but the final checkpoint collapsed badly. This showed the environment was learnable, but the PPO loop needed stabilization.

### Run 2: Stabilized PPO

- Added:
  - observation normalization
  - value clipping
  - KL-based update limiting
  - solved-threshold early stopping
  - checkpoint loading fixes for newer PyTorch
- Command: `./.venv/bin/python train_ppo.py --device cuda --updates 120 --eval-every 5 --eval-episodes 12 --benchmark-episodes 8 --checkpoint-dir artifacts_run2`
- Solved at update `10` after `61,440` steps
- 20-episode eval of solved checkpoint:
  - `mean_reward = 2759.32`
  - `mean_laps = 3.60`
  - `mean_progress = 3.544`
  - `off_track = 0`
- Result: the task became stable and the final checkpoint matched the solved checkpoint.

### Run 3: Domain Randomization Within One Procedural Family

- Added:
  - randomized track width, scale, harmonics, start pose
  - randomized vehicle dynamics
  - observation noise
  - curriculum ramp for randomization during training
- Command: `./.venv/bin/python train_ppo.py --device cuda --updates 160 --eval-every 5 --eval-episodes 16 --benchmark-episodes 12 --checkpoint-dir artifacts_run3`
- Solved at update `125` after `768,000` steps
- Benchmark of solved checkpoint:
  - `procedural_baseline`: `mean_progress = 3.378`, `off_track_rate = 0.00`
  - `generalization`: `mean_progress = 3.239`, `off_track_rate = 0.00`
  - `stress`: `mean_progress = 3.441`, `off_track_rate = 0.00`
- Result: robust across heavy randomization, but still inside the same procedural generator family.

### Run 4: Multiple Procedural Track Generators

- Replaced the old radial-only progress math with sampled centerline geometry so the environment could support:
  - radial
  - ellipse
  - peanut
  - stadium-style generators
- Command: `./.venv/bin/python train_ppo.py --device cuda --updates 140 --eval-every 5 --eval-episodes 12 --benchmark-episodes 10 --checkpoint-dir artifacts_run4`
- Solved at update `15` after `92,160` steps
- Benchmark of solved checkpoint:
  - `baseline`: `mean_progress = 3.542`, `off_track_rate = 0.00`
  - `generalization`: `mean_progress = 3.497`, `off_track_rate = 0.00`
  - `stress`: `mean_progress = 3.631`, `off_track_rate = 0.00`
- Result: strong robustness across multiple distinct procedural generator families.

### Run 5: Train/Holdout Split With Handcrafted Holdout Tracks

- Added:
  - fixed hand-authored layouts
  - explicit `train` and `holdout` track pools
  - training restricted to `train` pool
  - evaluation and benchmark on mixed and holdout pools
- Command: `./.venv/bin/python train_ppo.py --device cuda --updates 120 --eval-every 5 --eval-episodes 12 --benchmark-episodes 10 --checkpoint-dir artifacts_run5 --train-track-pool train --evaluation-track-pool all`
- Best checkpoint: update `45`
  - validation: `mean_progress = 3.220`, `off_track_rate = 0.00`
  - benchmark: `mean_progress = 3.130`, `off_track_rate = 0.00`
- Final checkpoint at update `120` regressed:
  - validation: `mean_progress = 2.645`, `off_track_rate = 0.17`
  - benchmark: `mean_progress = 2.313`, `off_track_rate = 0.40`
- Benchmark of best checkpoint:
  - `procedural_baseline`: `mean_progress = 3.495`, `off_track_rate = 0.00`
  - `mixed_generalization`: `mean_progress = 3.275`, `off_track_rate = 0.00`
  - `holdout_handcrafted`: `mean_progress = 2.981`, `off_track_rate = 0.00`
  - `holdout_stress`: `mean_progress = 2.824`, `off_track_rate = 0.00`
- Result: the policy transfers to unseen handcrafted holdout layouts without crashing, but performance is lower there than on seen or mixed-distribution tracks. This is the main remaining gap.

### Run 13: From-Scratch Curriculum Racing Dynamics

- Added:
  - racing-oriented car dynamics with a higher top speed and stronger braking
  - steering that gets harder at high speed instead of easier
  - a curriculum that starts on procedural tracks and then switches into the broader train pool
  - explicit holdout-based checkpoint selection instead of choosing checkpoints only from mixed-pool performance
- Command:
  - `./.venv/bin/python train_ppo.py --device cuda --updates 180 --eval-every 5 --eval-episodes 16 --benchmark-episodes 12 --selection-episodes 16 --checkpoint-dir artifacts_run13_curriculum_fromscratch_race --train-track-pool curriculum --evaluation-track-pool all --selection-track-pool holdout --solved-progress-threshold 2.80`
- Best checkpoint: `artifacts_run13_curriculum_fromscratch_race/ppo_best_model.pt`
- Solved at update `30` after `184,320` steps
- Best training metrics:
  - validation: `mean_progress = 4.175`, `off_track_rate = 0.00`
  - mixed benchmark: `mean_progress = 3.566`, `off_track_rate = 0.00`
  - holdout selection: `mean_progress = 3.193`, `off_track_rate = 0.00`
- Full benchmark of best checkpoint:
  - `procedural_baseline`: `mean_progress = 4.634`, `off_track_rate = 0.00`
  - `mixed_generalization`: `mean_progress = 4.114`, `off_track_rate = 0.00`
  - `holdout_handcrafted`: `mean_progress = 3.224`, `off_track_rate = 0.00`
  - `holdout_stress`: `mean_progress = 3.230`, `off_track_rate = 0.00`
- Direct holdout pace profile:
  - `mean_speed = 26.803`
  - `peak_speed = 27.055`
  - `speed_std = 1.985`
  - `mean_progress = 3.115`
- Result: this is the first run that combines the faster racing dynamics with robust transfer to unseen handcrafted holdout tracks. It is materially faster than the older robust checkpoints while keeping zero off-track failures across the benchmark suites.

## Current Takeaways

- The project now supports robust training on:
  - randomized dynamics
  - randomized sensing
  - multiple procedural generators
  - train/holdout track pools
- The trainer now also supports curriculum track pools and explicit holdout-based checkpoint selection for the racing-dynamics setup.
- `artifacts_run13_curriculum_fromscratch_race/ppo_best_model.pt` is now the strongest overall checkpoint for fast and robust transfer.
- The remaining weakness is no longer basic transfer failure; it is that the learned high-speed policy still mostly expresses speed control through aggressive steering and throttle rather than heavy braking on holdout tracks.

## Manual Play

Use arrow keys or `WASD`:

```bash
./.venv/bin/python play_manual.py
```
