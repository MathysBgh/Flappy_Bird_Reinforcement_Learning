from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import random
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import text_flappy_bird_gym


@dataclass(frozen=True)
class EnvConfig:
    env_id: str = "TextFlappyBird-v0"
    height: int = 15
    width: int = 20
    pipe_gap: int = 4


@dataclass(frozen=True)
class MonteCarloConfig:
    episodes: int = 2_000
    gamma: float = 0.99
    epsilon_start: float = 0.35
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8
    max_steps_per_episode: int = 500
    first_visit: bool = True
    evaluation_interval: int = 100
    evaluation_episodes: int = 25


@dataclass(frozen=True)
class SarsaLambdaConfig:
    episodes: int = 4_000
    alpha: float = 0.10
    gamma: float = 0.99
    lambda_value: float = 0.70
    epsilon_start: float = 0.30
    epsilon_end: float = 0.01
    epsilon_decay_fraction: float = 0.8
    max_steps_per_episode: int = 500
    evaluation_interval: int = 200
    evaluation_episodes: int = 25


@dataclass
class BaselineResult:
    scores: np.ndarray
    rewards: np.ndarray
    config: dict


@dataclass
class TrainingResult:
    agent_name: str
    q_values: np.ndarray
    training_scores: np.ndarray
    training_rewards: np.ndarray
    evaluation_episodes: np.ndarray
    evaluation_scores: np.ndarray
    evaluation_rewards: np.ndarray
    config: dict


@dataclass
class SweepResult:
    parameter_name: str
    values: np.ndarray
    mean_scores: np.ndarray
    std_scores: np.ndarray
    seeds: tuple[int, ...]


class StateAdapter:
    def __init__(self, env_config: EnvConfig):
        if env_config.env_id != "TextFlappyBird-v0":
            raise ValueError("Tabular agents in this module expect TextFlappyBird-v0.")

        self.height = env_config.height
        self.width = env_config.width
        self.pipe_gap = env_config.pipe_gap
        self.player_x = int(self.width * 0.3)

        self.x_min = 0
        self.x_max = self.width - 1 - self.player_x
        self.y_min = -(self.height - self.pipe_gap - 1 + self.pipe_gap // 2)
        self.y_max = self.height - 1 - self.pipe_gap // 2

        self.x_values = np.arange(self.x_min, self.x_max + 1, dtype=int)
        self.y_values = np.arange(self.y_min, self.y_max + 1, dtype=int)

    @property
    def q_shape(self) -> tuple[int, int, int]:
        return (len(self.x_values), len(self.y_values), 2)

    def state_to_index(self, state: tuple[int, int]) -> tuple[int, int]:
        x = int(np.clip(state[0], self.x_min, self.x_max))
        y = int(np.clip(state[1], self.y_min, self.y_max))
        return x - self.x_min, y - self.y_min

    def value_function(self, q_values: np.ndarray) -> np.ndarray:
        return np.max(q_values, axis=2)

    def greedy_policy(self, q_values: np.ndarray) -> np.ndarray:
        return np.argmax(q_values, axis=2)


def make_env(env_config: EnvConfig, seed: int | None = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = gym.make(
        env_config.env_id,
        height=env_config.height,
        width=env_config.width,
        pipe_gap=env_config.pipe_gap,
    )
    observation, info = env.reset()
    return env, observation, info


def linear_schedule(
    episode: int,
    total_episodes: int,
    start_value: float,
    end_value: float,
    decay_fraction: float,
) -> float:
    decay_steps = max(1, int(total_episodes * decay_fraction))
    progress = min(episode, decay_steps) / decay_steps
    return start_value + progress * (end_value - start_value)


def choose_epsilon_greedy_action(
    q_values: np.ndarray,
    state_index: tuple[int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[2]))

    state_action_values = q_values[state_index[0], state_index[1]]
    best_actions = np.flatnonzero(state_action_values == state_action_values.max())
    return int(rng.choice(best_actions))


def choose_greedy_action(
    q_values: np.ndarray,
    state_index: tuple[int, int],
    rng: np.random.Generator | None = None,
) -> int:
    state_action_values = q_values[state_index[0], state_index[1]]
    best_actions = np.flatnonzero(state_action_values == state_action_values.max())
    if rng is None or len(best_actions) == 1:
        return int(best_actions[0])
    return int(rng.choice(best_actions))


def run_random_baseline(
    env_config: EnvConfig,
    episodes: int = 100,
    max_steps_per_episode: int = 500,
    seed: int = 0,
) -> BaselineResult:
    scores = []
    rewards = []
    rng = np.random.default_rng(seed)

    for episode in range(episodes):
        env, observation, info = make_env(env_config, seed=seed + episode)
        total_reward = 0.0

        for _ in range(max_steps_per_episode):
            action = int(rng.integers(env.action_space.n))
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        scores.append(info["score"])
        rewards.append(total_reward)
        env.close()

    return BaselineResult(
        scores=np.asarray(scores, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        config={
            "episodes": episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "seed": seed,
        },
    )


def evaluate_policy(
    q_values: np.ndarray,
    evaluation_env_config: EnvConfig,
    state_adapter: StateAdapter,
    episodes: int = 50,
    max_steps_per_episode: int = 500,
    seed: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    rewards = []

    for episode in range(episodes):
        env, observation, info = make_env(evaluation_env_config, seed=seed + episode)
        total_reward = 0.0

        for _ in range(max_steps_per_episode):
            state_index = state_adapter.state_to_index(observation)
            action = choose_greedy_action(q_values, state_index)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        scores.append(info["score"])
        rewards.append(total_reward)
        env.close()

    return np.asarray(scores, dtype=float), np.asarray(rewards, dtype=float)


def train_monte_carlo_control(
    env_config: EnvConfig,
    train_config: MonteCarloConfig | None = None,
    seed: int = 0,
) -> TrainingResult:
    config = train_config or MonteCarloConfig()
    state_adapter = StateAdapter(env_config)
    q_values = np.zeros(state_adapter.q_shape, dtype=float)
    visit_counts = np.zeros(state_adapter.q_shape, dtype=np.int32)
    rng = np.random.default_rng(seed)

    training_scores = []
    training_rewards = []
    evaluation_episodes = []
    evaluation_scores = []
    evaluation_rewards = []

    for episode in range(config.episodes):
        epsilon = linear_schedule(
            episode=episode,
            total_episodes=config.episodes,
            start_value=config.epsilon_start,
            end_value=config.epsilon_end,
            decay_fraction=config.epsilon_decay_fraction,
        )
        env, observation, info = make_env(env_config, seed=seed + episode)
        trajectory = []
        total_reward = 0.0

        for _ in range(config.max_steps_per_episode):
            state_index = state_adapter.state_to_index(observation)
            action = choose_epsilon_greedy_action(q_values, state_index, epsilon, rng)
            next_observation, reward, done, truncated, info = env.step(action)
            trajectory.append((state_index, action, reward))
            total_reward += reward
            observation = next_observation
            if done or truncated:
                break

        training_scores.append(info["score"])
        training_rewards.append(total_reward)

        return_so_far = 0.0
        seen_state_actions = set()
        for state_index, action, reward in reversed(trajectory):
            return_so_far = reward + config.gamma * return_so_far
            key = (state_index[0], state_index[1], action)
            if config.first_visit and key in seen_state_actions:
                continue
            seen_state_actions.add(key)
            visit_counts[state_index[0], state_index[1], action] += 1
            alpha = 1.0 / visit_counts[state_index[0], state_index[1], action]
            q_values[state_index[0], state_index[1], action] += alpha * (
                return_so_far - q_values[state_index[0], state_index[1], action]
            )

        env.close()

        should_evaluate = (
            (episode + 1) % config.evaluation_interval == 0
            or episode == config.episodes - 1
        )
        if should_evaluate:
            eval_scores, eval_rewards = evaluate_policy(
                q_values=q_values,
                evaluation_env_config=env_config,
                state_adapter=state_adapter,
                episodes=config.evaluation_episodes,
                max_steps_per_episode=config.max_steps_per_episode,
                seed=seed + 50_000 + episode,
            )
            evaluation_episodes.append(episode + 1)
            evaluation_scores.append(eval_scores.mean())
            evaluation_rewards.append(eval_rewards.mean())

    return TrainingResult(
        agent_name="Monte Carlo Control",
        q_values=q_values,
        training_scores=np.asarray(training_scores, dtype=float),
        training_rewards=np.asarray(training_rewards, dtype=float),
        evaluation_episodes=np.asarray(evaluation_episodes, dtype=int),
        evaluation_scores=np.asarray(evaluation_scores, dtype=float),
        evaluation_rewards=np.asarray(evaluation_rewards, dtype=float),
        config=asdict(config),
    )


def train_sarsa_lambda(
    env_config: EnvConfig,
    train_config: SarsaLambdaConfig | None = None,
    seed: int = 0,
) -> TrainingResult:
    config = train_config or SarsaLambdaConfig()
    state_adapter = StateAdapter(env_config)
    q_values = np.zeros(state_adapter.q_shape, dtype=float)
    rng = np.random.default_rng(seed)

    training_scores = []
    training_rewards = []
    evaluation_episodes = []
    evaluation_scores = []
    evaluation_rewards = []

    for episode in range(config.episodes):
        epsilon = linear_schedule(
            episode=episode,
            total_episodes=config.episodes,
            start_value=config.epsilon_start,
            end_value=config.epsilon_end,
            decay_fraction=config.epsilon_decay_fraction,
        )
        env, observation, info = make_env(env_config, seed=seed + episode)
        eligibility_traces = np.zeros(state_adapter.q_shape, dtype=float)
        state_index = state_adapter.state_to_index(observation)
        action = choose_epsilon_greedy_action(q_values, state_index, epsilon, rng)
        total_reward = 0.0

        for _ in range(config.max_steps_per_episode):
            next_observation, reward, done, truncated, info = env.step(action)
            total_reward += reward

            eligibility_traces *= config.gamma * config.lambda_value
            eligibility_traces[state_index[0], state_index[1], action] += 1.0

            if done or truncated:
                delta = reward - q_values[state_index[0], state_index[1], action]
                q_values += config.alpha * delta * eligibility_traces
                break

            next_state_index = state_adapter.state_to_index(next_observation)
            next_action = choose_epsilon_greedy_action(
                q_values, next_state_index, epsilon, rng
            )
            delta = (
                reward
                + config.gamma
                * q_values[next_state_index[0], next_state_index[1], next_action]
                - q_values[state_index[0], state_index[1], action]
            )
            q_values += config.alpha * delta * eligibility_traces
            state_index = next_state_index
            action = next_action

        training_scores.append(info["score"])
        training_rewards.append(total_reward)
        env.close()

        should_evaluate = (
            (episode + 1) % config.evaluation_interval == 0
            or episode == config.episodes - 1
        )
        if should_evaluate:
            eval_scores, eval_rewards = evaluate_policy(
                q_values=q_values,
                evaluation_env_config=env_config,
                state_adapter=state_adapter,
                episodes=config.evaluation_episodes,
                max_steps_per_episode=config.max_steps_per_episode,
                seed=seed + 90_000 + episode,
            )
            evaluation_episodes.append(episode + 1)
            evaluation_scores.append(eval_scores.mean())
            evaluation_rewards.append(eval_rewards.mean())

    return TrainingResult(
        agent_name="Sarsa(lambda)",
        q_values=q_values,
        training_scores=np.asarray(training_scores, dtype=float),
        training_rewards=np.asarray(training_rewards, dtype=float),
        evaluation_episodes=np.asarray(evaluation_episodes, dtype=int),
        evaluation_scores=np.asarray(evaluation_scores, dtype=float),
        evaluation_rewards=np.asarray(evaluation_rewards, dtype=float),
        config=asdict(config),
    )


def moving_average(values: np.ndarray, window: int = 100) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.copy()
    if len(values) < window:
        return np.asarray([values.mean()] * len(values), dtype=float)
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    padding = np.full(window - 1, smoothed[0], dtype=float)
    return np.concatenate([padding, smoothed])


def plot_baseline(baseline: BaselineResult):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(baseline.scores, linewidth=1.5)
    axes[0].axhline(baseline.scores.mean(), color="tab:red", linestyle="--")
    axes[0].set_title("Random baseline scores")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")

    axes[1].plot(baseline.rewards, linewidth=1.5, color="tab:orange")
    axes[1].axhline(baseline.rewards.mean(), color="tab:red", linestyle="--")
    axes[1].set_title("Random baseline rewards")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total reward")

    plt.tight_layout()
    return fig, axes


def plot_training_comparison(
    monte_carlo_result: TrainingResult,
    sarsa_result: TrainingResult,
    baseline: BaselineResult | None = None,
    smoothing_window: int = 100,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    mc_scores = moving_average(monte_carlo_result.training_scores, smoothing_window)
    sarsa_scores = moving_average(sarsa_result.training_scores, smoothing_window)
    axes[0].plot(mc_scores, label="Monte Carlo", linewidth=2)
    axes[0].plot(sarsa_scores, label="Sarsa(lambda)", linewidth=2)
    if baseline is not None:
        axes[0].axhline(
            baseline.scores.mean(),
            color="black",
            linestyle="--",
            label="Random mean",
        )
    axes[0].set_title("Training score comparison")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")
    axes[0].legend()

    axes[1].plot(
        monte_carlo_result.evaluation_episodes,
        monte_carlo_result.evaluation_scores,
        marker="o",
        label="Monte Carlo eval",
        linewidth=2,
    )
    axes[1].plot(
        sarsa_result.evaluation_episodes,
        sarsa_result.evaluation_scores,
        marker="o",
        label="Sarsa(lambda) eval",
        linewidth=2,
    )
    if baseline is not None:
        axes[1].axhline(
            baseline.scores.mean(),
            color="black",
            linestyle="--",
            label="Random mean",
        )
    axes[1].set_title("Greedy evaluation score comparison")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average score")
    axes[1].legend()

    plt.tight_layout()
    return fig, axes


def plot_value_and_policy(
    q_values: np.ndarray,
    state_adapter: StateAdapter,
    title_prefix: str,
):
    values = state_adapter.value_function(q_values).T
    policy = state_adapter.greedy_policy(q_values).T

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    value_image = axes[0].imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[
            state_adapter.x_min - 0.5,
            state_adapter.x_max + 0.5,
            state_adapter.y_min - 0.5,
            state_adapter.y_max + 0.5,
        ],
        cmap="viridis",
    )
    axes[0].set_title(f"{title_prefix} value function")
    axes[0].set_xlabel("dx")
    axes[0].set_ylabel("dy")
    fig.colorbar(value_image, ax=axes[0], fraction=0.046, pad=0.04)

    policy_image = axes[1].imshow(
        policy,
        origin="lower",
        aspect="auto",
        extent=[
            state_adapter.x_min - 0.5,
            state_adapter.x_max + 0.5,
            state_adapter.y_min - 0.5,
            state_adapter.y_max + 0.5,
        ],
        cmap="coolwarm",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(f"{title_prefix} greedy policy")
    axes[1].set_xlabel("dx")
    axes[1].set_ylabel("dy")
    colorbar = fig.colorbar(policy_image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Idle", "Flap"])

    plt.tight_layout()
    return fig, axes


def plot_action_value_maps(
    q_values: np.ndarray,
    state_adapter: StateAdapter,
    title_prefix: str,
):
    idle_values = q_values[:, :, 0].T
    flap_values = q_values[:, :, 1].T
    vmin = float(np.min(q_values))
    vmax = float(np.max(q_values))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    idle_image = axes[0].imshow(
        idle_values,
        origin="lower",
        aspect="auto",
        extent=[
            state_adapter.x_min - 0.5,
            state_adapter.x_max + 0.5,
            state_adapter.y_min - 0.5,
            state_adapter.y_max + 0.5,
        ],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"{title_prefix} Q(dx, dy, Idle)")
    axes[0].set_xlabel("dx")
    axes[0].set_ylabel("dy")
    fig.colorbar(idle_image, ax=axes[0], fraction=0.046, pad=0.04)

    flap_image = axes[1].imshow(
        flap_values,
        origin="lower",
        aspect="auto",
        extent=[
            state_adapter.x_min - 0.5,
            state_adapter.x_max + 0.5,
            state_adapter.y_min - 0.5,
            state_adapter.y_max + 0.5,
        ],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"{title_prefix} Q(dx, dy, Flap)")
    axes[1].set_xlabel("dx")
    axes[1].set_ylabel("dy")
    fig.colorbar(flap_image, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig, axes


def summarize_result(result: TrainingResult) -> dict:
    return {
        "agent": result.agent_name,
        "best_training_score": float(result.training_scores.max()),
        "mean_last_100_training_scores": float(result.training_scores[-100:].mean()),
        "best_eval_score": float(result.evaluation_scores.max()),
        "final_eval_score": float(result.evaluation_scores[-1]),
    }


def parameter_sweep(
    train_function: Callable,
    base_config,
    parameter_name: str,
    values: list[float],
    env_config: EnvConfig,
    seeds: tuple[int, ...] = (0, 1, 2),
    evaluation_episodes: int = 50,
) -> SweepResult:
    state_adapter = StateAdapter(env_config)
    mean_scores = []
    std_scores = []

    for value in values:
        scores = []
        config = replace(base_config, **{parameter_name: value})
        for seed in seeds:
            result = train_function(env_config=env_config, train_config=config, seed=seed)
            eval_scores, _ = evaluate_policy(
                q_values=result.q_values,
                evaluation_env_config=env_config,
                state_adapter=state_adapter,
                episodes=evaluation_episodes,
                max_steps_per_episode=config.max_steps_per_episode,
                seed=seed + 200_000,
            )
            scores.append(eval_scores.mean())
        mean_scores.append(np.mean(scores))
        std_scores.append(np.std(scores))

    return SweepResult(
        parameter_name=parameter_name,
        values=np.asarray(values, dtype=float),
        mean_scores=np.asarray(mean_scores, dtype=float),
        std_scores=np.asarray(std_scores, dtype=float),
        seeds=seeds,
    )


def plot_parameter_sweep(sweep_result: SweepResult, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        sweep_result.values,
        sweep_result.mean_scores,
        yerr=sweep_result.std_scores,
        marker="o",
        capsize=4,
        linewidth=2,
    )
    ax.set_title(title)
    ax.set_xlabel(sweep_result.parameter_name)
    ax.set_ylabel("Average greedy evaluation score")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_transfer_comparison(
    source_env_config: EnvConfig,
    target_env_config: EnvConfig,
    monte_carlo_transfer: dict,
    sarsa_transfer: dict,
):
    labels = ["Train config", "Transfer config"]
    mc_scores = [
        monte_carlo_transfer["source_mean_score"],
        monte_carlo_transfer["target_mean_score"],
    ]
    sarsa_scores = [
        sarsa_transfer["source_mean_score"],
        sarsa_transfer["target_mean_score"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, mc_scores, width=width, label="Monte Carlo")
    ax.bar(x + width / 2, sarsa_scores, width=width, label="Sarsa(lambda)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average greedy evaluation score")
    ax.set_title(
        "Generalization across environment configurations\n"
        f"train: (h={source_env_config.height}, w={source_env_config.width}, gap={source_env_config.pipe_gap}) | "
        f"test: (h={target_env_config.height}, w={target_env_config.width}, gap={target_env_config.pipe_gap})"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, ax


def evaluate_transfer(
    q_values: np.ndarray,
    source_state_adapter: StateAdapter,
    source_env_config: EnvConfig,
    target_env_config: EnvConfig,
    episodes: int = 100,
    max_steps_per_episode: int = 500,
    seed: int = 300_000,
) -> dict:
    source_scores, source_rewards = evaluate_policy(
        q_values=q_values,
        evaluation_env_config=source_env_config,
        state_adapter=source_state_adapter,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
    )
    target_scores, target_rewards = evaluate_policy(
        q_values=q_values,
        evaluation_env_config=target_env_config,
        state_adapter=source_state_adapter,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed + 10_000,
    )
    return {
        "source_mean_score": float(source_scores.mean()),
        "target_mean_score": float(target_scores.mean()),
        "source_mean_reward": float(source_rewards.mean()),
        "target_mean_reward": float(target_rewards.mean()),
    }


def render_greedy_episode(
    q_values: np.ndarray,
    env_config: EnvConfig,
    state_adapter: StateAdapter,
    delay: float = 0.15,
    max_steps_per_episode: int = 200,
    seed: int = 400_000,
):
    from IPython.display import clear_output
    import time

    env, observation, info = make_env(env_config, seed=seed)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        print(env.render())
        print(f"step={step} | score={info['score']}")
        time.sleep(delay)

        state_index = state_adapter.state_to_index(observation)
        action = choose_greedy_action(q_values, state_index)
        observation, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break

    clear_output(wait=True)
    print(env.render())
    print(f"final score={info['score']}")
    env.close()
