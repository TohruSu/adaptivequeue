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


class Simulation:
    """Expose each ready queue and execute one job per scheduling decision."""

    def __init__(self, jobs: Iterable[Job]) -> None:
        self._arrivals = sorted(jobs, key=lambda job: (job.arrival_time, job.job_id))
        if len({job.job_id for job in self._arrivals}) != len(self._arrivals):
            raise ValueError("job IDs must be unique within an episode")
        self._waiting: list[Job] = []
        self._completed: list[JobResult] = []
        self._time = 0
        self._next_arrival = 0
        self._advance_to_decision()

    @property
    def waiting_jobs(self) -> tuple[Job, ...]:
        return tuple(self._waiting)

    @property
    def current_time(self) -> int:
        return self._time

    @property
    def done(self) -> bool:
        return not self._waiting

    def _advance_to_decision(self) -> None:
        if not self._waiting and self._next_arrival < len(self._arrivals):
            self._time = max(self._time, self._arrivals[self._next_arrival].arrival_time)
        while (
            self._next_arrival < len(self._arrivals)
            and self._arrivals[self._next_arrival].arrival_time <= self._time
        ):
            self._waiting.append(self._arrivals[self._next_arrival])
            self._next_arrival += 1

    def step(self, policy: Policy) -> JobResult:
        if self.done:
            raise ValueError("cannot step a completed simulation")
        selected = policy(self.waiting_jobs, self.current_time)
        if selected not in self._waiting:
            raise ValueError("policy must select a job from the waiting queue")
        self._waiting.remove(selected)
        record = JobResult(selected, self._time, self._time + selected.duration)
        self._completed.append(record)
        self._time = record.completion_time
        self._advance_to_decision()
        return record

    def result(self) -> SimulationResult:
        """Return execution records and metrics for jobs completed so far."""
        records = tuple(self._completed)
        return SimulationResult(records, calculate_metrics(records))


def simulate(jobs: Iterable[Job], policy: Policy) -> SimulationResult:
    """Run until every job finishes, returning records in execution order."""
    simulation = Simulation(jobs)
    while not simulation.done:
        simulation.step(policy)
    return simulation.result()
