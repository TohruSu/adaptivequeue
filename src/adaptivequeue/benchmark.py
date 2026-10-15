"""Paired scheduling experiments with separate Q-Learning training runs."""

import csv
import json
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean, stdev
from typing import Sequence

from adaptivequeue.metrics import Metrics
from adaptivequeue.models import Job
from adaptivequeue.policies import POLICY_NAMES, make_policy
from adaptivequeue.qlearning import (
    ACTION_NAMES,
    ACTION_POLICIES,
    LearningConfig,
    QLearningAgent,
    TrainingInfo,
    encode_state,
    train,
)
from adaptivequeue.simulation import simulate
from adaptivequeue.workloads import WORKLOADS, WorkloadConfig

TRAINING_SEEDS = (42, 2042, 4042)
EVALUATION_SEEDS = tuple(range(20000, 20030))
METRIC_NAMES = tuple(field.name for field in fields(Metrics))


@dataclass(frozen=True)
class EpisodeResult:
    workload: str
    evaluation_seed: int
    policy: str
    training_seed: int | None
    metrics: Metrics


def summarize(results: Sequence[EpisodeResult]) -> list[dict[str, object]]:
    """Average episode metrics and compare paired costs, without pooling RL runs."""
    groups: dict[tuple[str, str, int | None], list[EpisodeResult]] = {}
    baselines = {
        (row.workload, row.evaluation_seed, row.policy): row.metrics.primary_cost
        for row in results
        if row.training_seed is None
    }
    for row in results:
        groups.setdefault((row.workload, row.policy, row.training_seed), []).append(row)
    summary = []
    for (workload, policy, training_seed), episodes in groups.items():
        row_data: dict[str, object] = {
            "workload": workload,
            "policy": policy,
            "training_seed": training_seed,
            "episodes": len(episodes),
        }
        row_data.update(
            {
                f"mean_{name}": fmean(getattr(episode.metrics, name) for episode in episodes)
                for name in METRIC_NAMES
            }
        )
        row_data["std_primary_cost"] = (
            stdev(episode.metrics.primary_cost for episode in episodes)
            if len(episodes) > 1
            else 0.0
        )
        for baseline in POLICY_NAMES:
            differences = [
                episode.metrics.primary_cost
                - baselines[(workload, episode.evaluation_seed, baseline)]
                for episode in episodes
            ]
            row_data[f"mean_cost_delta_vs_{baseline}"] = fmean(differences)
            row_data[f"cost_win_rate_vs_{baseline}"] = fmean(value < 0 for value in differences)
        summary.append(row_data)
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_or_train(
    directory: Path, seeds: Sequence[int], episodes: int, *, retrain: bool
) -> dict[int, QLearningAgent]:
    directory.mkdir(parents=True, exist_ok=True)
    agents = {}
    for seed in seeds:
        path = directory / f"qlearning_seed_{seed}.json"
        if path.exists() and not retrain:
            agent = QLearningAgent.load(path)
            expected = TrainingInfo("mixed", WorkloadConfig(), episodes, seed)
            if agent.training != expected or agent.config != LearningConfig() or agent.seed != seed:
                raise ValueError(f"cached model configuration differs: {path}; use --retrain")
        else:
            agent = train(
                workload="mixed",
                episodes=episodes,
                seed=seed,
                workload_config=WorkloadConfig(),
                learning_config=LearningConfig(),
            )
            agent.save(path)
        agents[seed] = agent
    return agents


def _evaluate_agent(
    agent: QLearningAgent, jobs: Sequence[Job]
) -> tuple[Metrics, list[int], list[int]]:
    counts = [0] * len(ACTION_NAMES)
    contended = [0] * len(ACTION_NAMES)

    def policy(waiting_jobs: Sequence[Job], current_time: int) -> Job:
        action = agent.greedy_action(encode_state(waiting_jobs, current_time))
        counts[action] += 1
        contended[action] += len(waiting_jobs) > 1
        return ACTION_POLICIES[action](waiting_jobs, current_time)

    return simulate(jobs, policy).metrics, counts, contended


def plot_results(results: Sequence[EpisodeResult], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ValueError(
            "plotting requires Matplotlib; install adaptivequeue[benchmark]"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = list(dict.fromkeys(row.workload for row in results))
    series = list(dict.fromkeys((row.policy, row.training_seed) for row in results))
    labels = [name.upper() if seed is None else f"QL {seed}" for name, seed in series]
    for metric, filename, label in (
        ("primary_cost", "scheduling_cost.png", "Scheduling cost J (lower is better)"),
        ("deadline_miss_rate", "deadline_miss_rate.png", "Deadline miss rate"),
    ):
        figure, axes = plt.subplots(
            1, len(workloads), figsize=(5 * len(workloads), 4.8), squeeze=False
        )
        for axis, workload in zip(axes[0], workloads):
            values = [
                [
                    getattr(row.metrics, metric)
                    for row in results
                    if row.workload == workload and (row.policy, row.training_seed) == key
                ]
                for key in series
            ]
            boxes = axis.boxplot(values, showmeans=True, patch_artist=True)
            for index, box in enumerate(boxes["boxes"]):
                box.set_facecolor(plt.get_cmap("tab10")(index))
                box.set_alpha(0.55)
            axis.set_xticks(range(1, len(series) + 1), labels, rotation=45, ha="right")
            axis.set_title(workload)
            axis.set_ylabel(label)
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle("Episode distributions; line = median, triangle = mean")
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)


def run_benchmark(
    output_dir: str | Path,
    *,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
    training_episodes: int = 1000,
    evaluation_seeds: Sequence[int] = EVALUATION_SEEDS,
    workloads: Sequence[str] = ("stable", "bursty", "mixed"),
    retrain: bool = False,
    plots: bool = True,
) -> list[EpisodeResult]:
    """Run the fixed protocol; smaller seed sets are useful for focused tests."""
    if training_episodes < 1 or not training_seeds or not evaluation_seeds or not workloads:
        raise ValueError(
            "benchmark needs positive training episodes and nonempty seed/workload sets"
        )
    for seeds in (training_seeds, evaluation_seeds):
        if len(set(seeds)) != len(seeds) or min(seeds) < 0:
            raise ValueError("benchmark seeds must be unique and nonnegative")
    if any(name not in WORKLOADS for name in workloads) or len(set(workloads)) != len(workloads):
        raise ValueError("benchmark workloads must be unique registered workload names")
    if any(
        start <= seed < start + training_episodes
        for start in training_seeds
        for seed in evaluation_seeds
    ):
        raise ValueError("evaluation seeds must not overlap training seeds")

    directory = Path(output_dir)
    agents = _load_or_train(
        directory / "models", training_seeds, training_episodes, retrain=retrain
    )
    results = []
    counts = {
        (workload, seed): [0] * len(ACTION_NAMES)
        for workload in workloads
        for seed in training_seeds
    }
    contended = {
        (workload, seed): [0] * len(ACTION_NAMES)
        for workload in workloads
        for seed in training_seeds
    }
    for workload in workloads:
        for evaluation_seed in evaluation_seeds:
            jobs = WORKLOADS[workload](evaluation_seed, WorkloadConfig())
            for name in POLICY_NAMES:
                metrics = simulate(jobs, make_policy(name, seed=evaluation_seed)).metrics
                results.append(EpisodeResult(workload, evaluation_seed, name, None, metrics))
            for training_seed, agent in agents.items():
                metrics, selections, choices = _evaluate_agent(agent, jobs)
                results.append(
                    EpisodeResult(workload, evaluation_seed, "qlearning", training_seed, metrics)
                )
                for action in range(len(ACTION_NAMES)):
                    counts[(workload, training_seed)][action] += selections[action]
                    contended[(workload, training_seed)][action] += choices[action]

    episode_rows = [
        {
            "workload": row.workload,
            "evaluation_seed": row.evaluation_seed,
            "policy": row.policy,
            "training_seed": row.training_seed,
            **asdict(row.metrics),
        }
        for row in results
    ]
    action_rows: list[dict[str, object]] = []
    for (workload, seed), selections in counts.items():
        choices = contended[(workload, seed)]
        total, total_contended = sum(selections), sum(choices)
        for action, name in enumerate(ACTION_NAMES):
            action_rows.append(
                {
                    "workload": workload,
                    "training_seed": seed,
                    "action": action,
                    "policy": name,
                    "selections": selections[action],
                    "fraction": selections[action] / total if total else 0.0,
                    "contended_selections": choices[action],
                    "contended_fraction": choices[action] / total_contended
                    if total_contended
                    else 0.0,
                }
            )
    _write_csv(directory / "per_episode.csv", episode_rows)
    _write_csv(directory / "summary.csv", summarize(results))
    _write_csv(directory / "action_distribution.csv", action_rows)
    protocol = {
        "training_workload": "mixed",
        "training_episodes": training_episodes,
        "training_seeds": list(training_seeds),
        "learning_config": asdict(LearningConfig()),
        "workload_config": asdict(WorkloadConfig()),
        "evaluation_workloads": list(workloads),
        "evaluation_seeds": list(evaluation_seeds),
        "random_policy_seed": "evaluation_seed",
        "aggregation": "mean of episode metrics; sample standard deviation of episode cost",
        "python_version": platform.python_version(),
    }
    (directory / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    if plots:
        plot_results(results, directory / "figures")
    return results
