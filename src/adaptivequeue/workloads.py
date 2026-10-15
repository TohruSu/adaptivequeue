"""Seeded synthetic workloads with configurable load and deadline budgets."""

import random
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Callable

from adaptivequeue.models import Job


@dataclass(frozen=True)
class WorkloadConfig:
    num_jobs: int = 60
    arrival_rate: float = 0.28
    deadline_factor: float = 4.0
    duration_range: tuple[int, int] = (1, 6)
    burst_size: int = 6
    burst_window: int = 3

    def __post_init__(self) -> None:
        if self.num_jobs < 0:
            raise ValueError("num_jobs must be nonnegative")
        if not isfinite(self.arrival_rate) or self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be finite and positive")
        if not isfinite(self.deadline_factor) or self.deadline_factor <= 0:
            raise ValueError("deadline_factor must be finite and positive")
        low, high = self.duration_range
        if low < 1 or high < low:
            raise ValueError("duration_range must satisfy 1 <= minimum <= maximum")
        if self.burst_size < 1 or self.burst_window < 1:
            raise ValueError("burst_size and burst_window must be positive")


def _steady_arrivals(rng: random.Random, config: WorkloadConfig) -> list[int]:
    time = 0.0
    arrivals: list[int] = []
    for _ in range(config.num_jobs):
        arrivals.append(int(time))
        time += rng.expovariate(config.arrival_rate)
    return arrivals


def _make_jobs(
    arrivals: list[int], rng: random.Random, config: WorkloadConfig, *, mixed: bool = False
) -> list[Job]:
    jobs: list[Job] = []
    for index, arrival in enumerate(sorted(arrivals)):
        duration = rng.randint(*config.duration_range)
        if mixed:
            job_type, weight, factor = rng.choice(
                (("interactive", 4, 0.5), ("report", 2, 1.0), ("maintenance", 1, 2.0))
            )
        else:
            job_type = rng.choice(("email", "image", "backup"))
            weight = rng.choice((1, 2, 4))
            factor = 1.0
        budget = max(1, ceil(duration * config.deadline_factor * factor * rng.uniform(0.75, 1.25)))
        jobs.append(Job(f"job-{index:03d}", arrival, duration, arrival + budget, weight, job_type))
    return jobs


def stable_arrivals(seed: int, config: WorkloadConfig = WorkloadConfig()) -> list[Job]:
    """Constant-rate arrivals, rounded down to integer time units."""
    rng = random.Random(seed)
    return _make_jobs(_steady_arrivals(rng, config), rng, config)


def bursty_arrivals(seed: int, config: WorkloadConfig = WorkloadConfig()) -> list[Job]:
    """Batches arriving within short windows, separated by random gaps."""
    rng = random.Random(seed)
    time = 0.0
    arrivals: list[int] = []
    for offset in range(0, config.num_jobs, config.burst_size):
        size = min(config.burst_size, config.num_jobs - offset)
        arrivals.extend(int(time) + rng.randrange(config.burst_window) for _ in range(size))
        time += rng.expovariate(config.arrival_rate / config.burst_size)
    return _make_jobs(arrivals, rng, config)


def mixed_workload(seed: int, config: WorkloadConfig = WorkloadConfig()) -> list[Job]:
    """Constant-rate arrivals with tight/high-weight and loose/low-weight jobs."""
    rng = random.Random(seed)
    return _make_jobs(_steady_arrivals(rng, config), rng, config, mixed=True)


WORKLOADS: dict[str, Callable[[int, WorkloadConfig], list[Job]]] = {
    "stable": stable_arrivals,
    "bursty": bursty_arrivals,
    "mixed": mixed_workload,
}
