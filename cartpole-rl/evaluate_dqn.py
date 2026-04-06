from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from cartpole_rl.dqn import QNetwork, infer_q_hidden_dims
from cartpole_rl.environment import CartPoleEnv


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DQN CartPole agent.")
    parser.add_argument("--checkpoint", type=str, default="dqn_artifacts/dqn_best_model.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--render", choices=("none", "text", "human"), default="none")
    parser.add_argument("--delay", type=float, default=0.03)
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    max_steps = int(args.max_steps or checkpoint_config.get("max_steps", 500))
    hidden_dims = infer_q_hidden_dims(checkpoint["q_network"])

    q_network = QNetwork(CartPoleEnv.observation_size, CartPoleEnv.action_size, hidden_dims=hidden_dims).to(device)
    q_network.load_state_dict(checkpoint["q_network"])
    q_network.eval()

    returns = []
    lengths = []
    render_enabled = args.render in {"text", "human"}

    if device.type == "cuda":
        print(f"Evaluating on GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Evaluating on CPU")

    for episode in range(1, args.episodes + 1):
        env = CartPoleEnv(seed=args.seed + episode, max_steps=max_steps)
        state, _ = env.reset(seed=args.seed + episode)
        episode_return = 0.0

        for _ in range(max_steps):
            with torch.no_grad():
                q_values = q_network(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(torch.argmax(q_values, dim=1).item())

            step = env.step(action)
            state = step.observation
            episode_return += step.reward

            if render_enabled:
                print("\033[2J\033[H", end="")
                print(env.render_text())
                print(f"episode={episode} reward={episode_return:0.1f} action={action}")
                time.sleep(args.delay)

            if step.terminated or step.truncated:
                returns.append(episode_return)
                lengths.append(int(step.info["steps"]))
                end_reason = "truncated" if step.truncated and not step.terminated else "terminated"
                print(
                    f"episode={episode} return={episode_return:0.1f} "
                    f"steps={int(step.info['steps'])} end={end_reason}"
                )
                break

        env.close()

    solved_rate = float(np.mean([value >= max_steps for value in returns])) if returns else 0.0
    print(
        f"mean_return={np.mean(returns):0.2f} median_return={np.median(returns):0.2f} "
        f"mean_length={np.mean(lengths):0.2f} solved_rate={solved_rate:0.2f}"
    )


if __name__ == "__main__":
    main()
