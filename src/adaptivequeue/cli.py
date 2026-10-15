"""Baseline episodes and tabular Q-Learning training/evaluation."""

import argparse
from pathlib import Path

from adaptivequeue.metrics import Metrics
from adaptivequeue.policies import POLICY_NAMES, make_policy
from adaptivequeue.qlearning import LearningConfig, QLearningAgent, evaluate, train
from adaptivequeue.simulation import simulate
from adaptivequeue.workloads import WORKLOADS, WorkloadConfig


def _add_workload_arguments(
    parser: argparse.ArgumentParser, *, workload: str = "stable", seed: int | None = 42
) -> None:
    parser.add_argument("--workload", choices=WORKLOADS, default=workload)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--arrival-rate", type=float, default=0.28, help="Mean jobs per time unit")
    parser.add_argument(
        "--deadline-factor", type=float, default=4.0, help="Larger means looser deadlines"
    )
    parser.add_argument("--burst-size", type=int, default=6)
    parser.add_argument(
        "--burst-window", type=int, default=3, help="Burst arrival window in time units"
    )


def _print_metrics(metrics: Metrics) -> None:
    print(f"Primary cost J (lower is better): {metrics.primary_cost:.3f}")
    print(f"Deadline miss rate: {metrics.deadline_miss_rate:.1%}")
    print(f"Average waiting time: {metrics.average_waiting_time:.3f}")
    print(f"95th-percentile waiting time: {metrics.p95_waiting_time:.3f}")
    print(f"Average flow time: {metrics.average_flow_time:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AdaptiveQueue scheduling experiments.")
    _add_workload_arguments(parser)
    parser.add_argument("--policy", choices=POLICY_NAMES, default="fifo")
    commands = parser.add_subparsers(dest="command")

    training = commands.add_parser("train", help="Train and save a Q-Learning meta-scheduler")
    _add_workload_arguments(training, workload="mixed")
    training.add_argument("--episodes", type=int, default=1000)
    training.add_argument("--learning-rate", type=float, default=0.1)
    training.add_argument("--gamma", type=float, default=1.0)
    training.add_argument("--epsilon-start", type=float, default=1.0)
    training.add_argument("--epsilon-end", type=float, default=0.05)
    training.add_argument("--epsilon-decay", type=float, default=0.995)
    training.add_argument("--output", type=Path, required=True)

    evaluation = commands.add_parser("evaluate", help="Evaluate a frozen Q-table on held-out seeds")
    _add_workload_arguments(evaluation, workload="mixed", seed=None)
    evaluation.add_argument("--episodes", type=int, default=20)
    evaluation.add_argument("--model", type=Path, required=True)

    benchmark = commands.add_parser("benchmark", help="Run the reproducible paired benchmark")
    benchmark.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    benchmark.add_argument("--retrain", action="store_true", help="Replace cached benchmark models")
    args = parser.parse_args()

    try:
        if args.command == "benchmark":
            from adaptivequeue.benchmark import run_benchmark

            results = run_benchmark(args.output_dir, retrain=args.retrain)
            print(
                f"Saved {len(results)} per-episode results, "
                f"summaries, and figures to {args.output_dir}"
            )
            return
        config = WorkloadConfig(
            num_jobs=args.jobs,
            arrival_rate=args.arrival_rate,
            deadline_factor=args.deadline_factor,
            burst_size=args.burst_size,
            burst_window=args.burst_window,
        )
        if args.command == "train":
            learning = LearningConfig(
                args.learning_rate,
                args.gamma,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay,
            )
            agent = train(
                workload=args.workload,
                episodes=args.episodes,
                seed=args.seed,
                workload_config=config,
                learning_config=learning,
            )
            agent.save(args.output)
            print(
                f"Trained {args.episodes} episodes; "
                f"{len(agent.q_table)} states; saved {args.output}"
            )
        elif args.command == "evaluate":
            agent = QLearningAgent.load(args.model)
            metrics = evaluate(
                agent,
                workload=args.workload,
                episodes=args.episodes,
                seed=args.seed,
                workload_config=config,
            )
            print(f"Workload: {args.workload} | Policy: qlearning | Episodes: {args.episodes}")
            _print_metrics(metrics)
        else:
            jobs = WORKLOADS[args.workload](args.seed, config)
            result = simulate(jobs, make_policy(args.policy, seed=args.seed))
            print(
                f"Workload: {args.workload} | Policy: {args.policy} | "
                f"Seed: {args.seed} | Jobs: {len(jobs)}"
            )
            _print_metrics(result.metrics)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
