from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from flappy_bird import (
    EnvConfig,
    MonteCarloConfig,
    SarsaLambdaConfig,
    StateAdapter,
    evaluate_transfer,
    parameter_sweep,
    plot_training_comparison,
    plot_transfer_comparison,
    run_random_baseline,
    summarize_result,
    train_monte_carlo_control,
    train_sarsa_lambda,
)


STUDENT_NAME = "Mathys Bagnah"
REPO_URL = "https://github.com/MathysBgh/Flappy_Bird_Reinforcement_Learning"


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def save_value_action_summary(
    output_path: Path,
    state_adapter: StateAdapter,
    monte_carlo_q: np.ndarray,
    sarsa_q: np.ndarray,
) -> None:
    mc_v = state_adapter.value_function(monte_carlo_q).T
    sarsa_v = state_adapter.value_function(sarsa_q).T
    mc_q_flap = monte_carlo_q[:, :, 1].T
    sarsa_q_flap = sarsa_q[:, :, 1].T

    v_min = float(min(mc_v.min(), sarsa_v.min()))
    v_max = float(max(mc_v.max(), sarsa_v.max()))
    q_min = float(min(mc_q_flap.min(), sarsa_q_flap.min()))
    q_max = float(max(mc_q_flap.max(), sarsa_q_flap.max()))

    extent = [
        state_adapter.x_min - 0.5,
        state_adapter.x_max + 0.5,
        state_adapter.y_min - 0.5,
        state_adapter.y_max + 0.5,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))

    im00 = axes[0, 0].imshow(mc_v, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=v_min, vmax=v_max)
    axes[0, 0].set_title("Monte Carlo $V(s)$")
    axes[0, 0].set_xlabel("dx")
    axes[0, 0].set_ylabel("dy")

    im01 = axes[0, 1].imshow(sarsa_v, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=v_min, vmax=v_max)
    axes[0, 1].set_title("Sarsa($\\lambda$) $V(s)$")
    axes[0, 1].set_xlabel("dx")
    axes[0, 1].set_ylabel("dy")

    im10 = axes[1, 0].imshow(mc_q_flap, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=q_min, vmax=q_max)
    axes[1, 0].set_title("Monte Carlo $Q(s,\\mathrm{Flap})$")
    axes[1, 0].set_xlabel("dx")
    axes[1, 0].set_ylabel("dy")

    im11 = axes[1, 1].imshow(sarsa_q_flap, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=q_min, vmax=q_max)
    axes[1, 1].set_title("Sarsa($\\lambda$) $Q(s,\\mathrm{Flap})$")
    axes[1, 1].set_xlabel("dx")
    axes[1, 1].set_ylabel("dy")

    fig.colorbar(im01, ax=axes[0, :], fraction=0.03, pad=0.02)
    fig.colorbar(im11, ax=axes[1, :], fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sweep_summary(
    output_path: Path,
    mc_sweep,
    sarsa_sweep,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    axes[0].errorbar(
        mc_sweep.values,
        mc_sweep.mean_scores,
        yerr=mc_sweep.std_scores,
        marker="o",
        capsize=4,
        linewidth=2,
        color="#1f77b4",
    )
    axes[0].set_title("Monte Carlo sweep")
    axes[0].set_xlabel("epsilon_end")
    axes[0].set_ylabel("Average greedy evaluation score")
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(
        sarsa_sweep.values,
        sarsa_sweep.mean_scores,
        yerr=sarsa_sweep.std_scores,
        marker="o",
        capsize=4,
        linewidth=2,
        color="#ff7f0e",
    )
    axes[1].set_title("Sarsa($\\lambda$) sweep")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("Average greedy evaluation score")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_training_summary_column(
    output_path: Path,
    baseline,
    monte_carlo_result,
    sarsa_result,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 6.4))

    window = 100
    def moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < window:
            return np.asarray([values.mean()] * len(values))
        kernel = np.ones(window, dtype=float) / window
        smooth = np.convolve(values, kernel, mode="valid")
        return np.concatenate([np.full(window - 1, smooth[0]), smooth])

    axes[0].plot(moving_average(monte_carlo_result.training_scores, window), linewidth=2, label="Monte Carlo")
    axes[0].plot(moving_average(sarsa_result.training_scores, window), linewidth=2, label="Sarsa($\\lambda$)")
    axes[0].axhline(baseline.scores.mean(), linestyle="--", color="black", linewidth=1.5, label="Random mean")
    axes[0].set_title("Training scores")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(monte_carlo_result.evaluation_episodes, monte_carlo_result.evaluation_scores, marker="o", linewidth=2, label="Monte Carlo")
    axes[1].plot(sarsa_result.evaluation_episodes, sarsa_result.evaluation_scores, marker="o", linewidth=2, label="Sarsa($\\lambda$)")
    axes[1].axhline(baseline.scores.mean(), linestyle="--", color="black", linewidth=1.5, label="Random mean")
    axes[1].set_title("Greedy evaluation")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average score")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_parameter_transfer_column(
    output_path: Path,
    mc_sweep,
    sarsa_sweep,
    mc_transfer: dict,
    sarsa_transfer: dict,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 8.2))

    axes[0].errorbar(mc_sweep.values, mc_sweep.mean_scores, yerr=mc_sweep.std_scores, marker="o", capsize=4, linewidth=2, color="#1f77b4")
    axes[0].set_title("Monte Carlo sweep")
    axes[0].set_xlabel("epsilon_end")
    axes[0].set_ylabel("Eval score")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(sarsa_sweep.values, sarsa_sweep.mean_scores, yerr=sarsa_sweep.std_scores, marker="o", capsize=4, linewidth=2, color="#ff7f0e")
    axes[1].set_title("Sarsa($\\lambda$) sweep")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("Eval score")
    axes[1].grid(alpha=0.25)

    labels = ["Train", "Transfer"]
    x = np.arange(len(labels))
    width = 0.35
    axes[2].bar(x - width / 2, [mc_transfer["source_mean_score"], mc_transfer["target_mean_score"]], width=width, label="Monte Carlo")
    axes[2].bar(x + width / 2, [sarsa_transfer["source_mean_score"], sarsa_transfer["target_mean_score"]], width=width, label="Sarsa($\\lambda$)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Configuration transfer")
    axes[2].set_ylabel("Eval score")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_metrics_file(
    output_path: Path,
    baseline,
    mc_summary: dict,
    sarsa_summary: dict,
    mc_transfer: dict,
    sarsa_transfer: dict,
) -> None:
    output_path.write_text(
        "\n".join(
            [
                rf"\newcommand{{\StudentName}}{{{STUDENT_NAME}}}",
                rf"\newcommand{{\RepoURL}}{{{REPO_URL}}}",
                r"\newcommand{\TrainHeight}{15}",
                r"\newcommand{\TrainWidth}{20}",
                r"\newcommand{\TrainGap}{4}",
                r"\newcommand{\TransferHeight}{17}",
                r"\newcommand{\TransferWidth}{22}",
                r"\newcommand{\TransferGap}{5}",
                rf"\newcommand{{\BaselineMeanScore}}{{{fmt(baseline.scores.mean())}}}",
                rf"\newcommand{{\BaselineBestScore}}{{{int(baseline.scores.max())}}}",
                rf"\newcommand{{\BaselineMeanReward}}{{{fmt(baseline.rewards.mean())}}}",
                rf"\newcommand{{\MCFinalEval}}{{{fmt(mc_summary['final_eval_score'])}}}",
                rf"\newcommand{{\MCBestTrain}}{{{fmt(mc_summary['best_training_score'])}}}",
                rf"\newcommand{{\MCLastHundred}}{{{fmt(mc_summary['mean_last_100_training_scores'])}}}",
                rf"\newcommand{{\SarsaFinalEval}}{{{fmt(sarsa_summary['final_eval_score'])}}}",
                rf"\newcommand{{\SarsaBestTrain}}{{{fmt(sarsa_summary['best_training_score'])}}}",
                rf"\newcommand{{\SarsaLastHundred}}{{{fmt(sarsa_summary['mean_last_100_training_scores'])}}}",
                rf"\newcommand{{\MCTransferScore}}{{{fmt(mc_transfer['target_mean_score'])}}}",
                rf"\newcommand{{\SarsaTransferScore}}{{{fmt(sarsa_transfer['target_mean_score'])}}}",
                rf"\newcommand{{\MCTrainScore}}{{{fmt(mc_transfer['source_mean_score'])}}}",
                rf"\newcommand{{\SarsaTrainScore}}{{{fmt(sarsa_transfer['source_mean_score'])}}}",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    report_dir = Path("report")
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    env_config = EnvConfig(env_id="TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    state_adapter = StateAdapter(env_config)

    baseline = run_random_baseline(env_config, episodes=100, max_steps_per_episode=500, seed=0)
    mc_result = train_monte_carlo_control(env_config, MonteCarloConfig(), seed=0)
    sarsa_result = train_sarsa_lambda(env_config, SarsaLambdaConfig(), seed=0)

    mc_summary = summarize_result(mc_result)
    sarsa_summary = summarize_result(sarsa_result)

    mc_sweep = parameter_sweep(
        train_function=train_monte_carlo_control,
        base_config=MonteCarloConfig(
            episodes=800,
            gamma=0.99,
            epsilon_start=0.35,
            epsilon_end=0.05,
            epsilon_decay_fraction=0.8,
            max_steps_per_episode=500,
            first_visit=True,
            evaluation_interval=400,
            evaluation_episodes=20,
        ),
        parameter_name="epsilon_end",
        values=[0.01, 0.05, 0.10],
        env_config=env_config,
        seeds=(0, 1),
        evaluation_episodes=30,
    )
    sarsa_sweep = parameter_sweep(
        train_function=train_sarsa_lambda,
        base_config=SarsaLambdaConfig(
            episodes=1600,
            alpha=0.10,
            gamma=0.99,
            lambda_value=0.70,
            epsilon_start=0.30,
            epsilon_end=0.01,
            epsilon_decay_fraction=0.8,
            max_steps_per_episode=500,
            evaluation_interval=400,
            evaluation_episodes=20,
        ),
        parameter_name="lambda_value",
        values=[0.50, 0.70, 0.90],
        env_config=env_config,
        seeds=(0, 1),
        evaluation_episodes=30,
    )

    transfer_config = EnvConfig(env_id="TextFlappyBird-v0", height=17, width=22, pipe_gap=5)
    mc_transfer = evaluate_transfer(
        q_values=mc_result.q_values,
        source_state_adapter=state_adapter,
        source_env_config=env_config,
        target_env_config=transfer_config,
        episodes=100,
        max_steps_per_episode=500,
        seed=12345,
    )
    sarsa_transfer = evaluate_transfer(
        q_values=sarsa_result.q_values,
        source_state_adapter=state_adapter,
        source_env_config=env_config,
        target_env_config=transfer_config,
        episodes=100,
        max_steps_per_episode=500,
        seed=54321,
    )

    fig, _ = plot_training_comparison(mc_result, sarsa_result, baseline=baseline, smoothing_window=100)
    fig.savefig(figures_dir / "training_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_training_summary_column(figures_dir / "training_comparison_column.png", baseline, mc_result, sarsa_result)

    save_value_action_summary(figures_dir / "value_action_summary.png", state_adapter, mc_result.q_values, sarsa_result.q_values)
    save_sweep_summary(figures_dir / "parameter_sweeps.png", mc_sweep, sarsa_sweep)

    fig, _ = plot_transfer_comparison(env_config, transfer_config, mc_transfer, sarsa_transfer)
    fig.savefig(figures_dir / "transfer_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_parameter_transfer_column(figures_dir / "parameter_transfer_column.png", mc_sweep, sarsa_sweep, mc_transfer, sarsa_transfer)

    write_metrics_file(report_dir / "metrics.tex", baseline, mc_summary, sarsa_summary, mc_transfer, sarsa_transfer)

    print("Assets generated in", report_dir)
    print("Monte Carlo summary:", mc_summary)
    print("Sarsa(lambda) summary:", sarsa_summary)
    print("Transfer results:", {"mc": mc_transfer, "sarsa": sarsa_transfer})


if __name__ == "__main__":
    main()
