"""A single-worker, non-preemptive simulation on integer time boundaries."""

from dataclasses import dataclass
from typing import Iterable

from adaptivequeue.metrics import Metrics, calculate_metrics
from adaptivequeue.models import Job, JobResult
from adaptivequeue.policies import Policy


@dataclass(frozen=True)
class SimulationResult:
    jobs: tuple[JobResult, ...]
    metrics: Metrics


def simulate(jobs: Iterable[Job], policy: Policy) -> SimulationResult:
    """Run until every job finishes, returning records in execution order."""
    arrivals = sorted(jobs, key=lambda job: (job.arrival_time, job.job_id))
    if len({job.job_id for job in arrivals}) != len(arrivals):
        raise ValueError("job IDs must be unique within an episode")

    waiting: list[Job] = []
    completed: list[JobResult] = []
    time = 0
    next_arrival = 0

    while waiting or next_arrival < len(arrivals):
        if not waiting:
            time = max(time, arrivals[next_arrival].arrival_time)
        while next_arrival < len(arrivals) and arrivals[next_arrival].arrival_time <= time:
            waiting.append(arrivals[next_arrival])
            next_arrival += 1

        selected = policy(tuple(waiting), time)
        if selected not in waiting:
            raise ValueError("policy must select a job from the waiting queue")
        waiting.remove(selected)
        completion_time = time + selected.duration
        completed.append(JobResult(selected, time, completion_time))
        time = completion_time

    records = tuple(completed)
    return SimulationResult(records, calculate_metrics(records))
