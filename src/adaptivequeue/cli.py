"""Command-line entry point for one reproducible scheduling episode."""

import argparse

from adaptivequeue.policies import POLICY_NAMES, make_policy
from adaptivequeue.simulation import simulate
from adaptivequeue.workloads import WORKLOADS, WorkloadConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AdaptiveQueue scheduling episode.")
    parser.add_argument("--workload", choices=WORKLOADS, default="stable")
    parser.add_argument("--policy", choices=POLICY_NAMES, default="fifo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--arrival-rate", type=float, default=0.28, help="Mean jobs per time unit")
    parser.add_argument(
        "--deadline-factor", type=float, default=4.0, help="Larger means looser deadlines"
    )
    parser.add_argument("--burst-size", type=int, default=6)
    parser.add_argument(
        "--burst-window", type=int, default=3, help="Burst arrival window in time units"
    )
    args = parser.parse_args()

    try:
        config = WorkloadConfig(
            num_jobs=args.jobs,
            arrival_rate=args.arrival_rate,
            deadline_factor=args.deadline_factor,
            burst_size=args.burst_size,
            burst_window=args.burst_window,
        )
    except ValueError as error:
        parser.error(str(error))

    jobs = WORKLOADS[args.workload](args.seed, config)
    result = simulate(jobs, make_policy(args.policy, seed=args.seed))
    metrics = result.metrics
    print(
        f"Workload: {args.workload} | Policy: {args.policy} | Seed: {args.seed} | Jobs: {len(jobs)}"
    )
    print(f"Primary cost J (lower is better): {metrics.primary_cost:.3f}")
    print(f"Deadline miss rate: {metrics.deadline_miss_rate:.1%}")
    print(f"Average waiting time: {metrics.average_waiting_time:.3f}")
    print(f"95th-percentile waiting time: {metrics.p95_waiting_time:.3f}")
    print(f"Average flow time: {metrics.average_flow_time:.3f}")
