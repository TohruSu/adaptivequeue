"""Episode metrics; the waiting-time percentile uses linear interpolation."""

from dataclasses import dataclass
from math import ceil, floor
from statistics import fmean
from typing import Sequence

from adaptivequeue.models import JobResult


@dataclass(frozen=True)
class Metrics:
    primary_cost: float
    deadline_miss_rate: float
    average_waiting_time: float
    p95_waiting_time: float
    average_flow_time: float


def calculate_metrics(results: Sequence[JobResult]) -> Metrics:
    if not results:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0)

    waiting = sorted(result.waiting_time for result in results)
    position = 0.95 * (len(waiting) - 1)
    lower, upper = floor(position), ceil(position)
    p95 = waiting[lower] + (waiting[upper] - waiting[lower]) * (position - lower)
    return Metrics(
        primary_cost=fmean(
            result.job.priority_weight * (result.tardiness + 0.1 * result.flow_time)
            for result in results
        ),
        deadline_miss_rate=fmean(result.tardiness > 0 for result in results),
        average_waiting_time=fmean(waiting),
        p95_waiting_time=p95,
        average_flow_time=fmean(result.flow_time for result in results),
    )
