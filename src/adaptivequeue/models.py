"""Immutable job specifications and execution records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    job_id: str
    arrival_time: int
    duration: int
    deadline: int
    priority_weight: int
    job_type: str

    def __post_init__(self) -> None:
        if not self.job_id or not self.job_type:
            raise ValueError("job_id and job_type must be nonempty")
        if any(
            not isinstance(value, int)
            for value in (self.arrival_time, self.duration, self.deadline, self.priority_weight)
        ):
            raise ValueError("job times, duration, and priority_weight must be integers")
        if self.arrival_time < 0 or self.deadline < 0:
            raise ValueError("arrival_time and deadline must be nonnegative")
        if self.duration <= 0 or self.priority_weight <= 0:
            raise ValueError("duration and priority_weight must be positive")


@dataclass(frozen=True)
class JobResult:
    job: Job
    start_time: int
    completion_time: int

    @property
    def waiting_time(self) -> int:
        return self.start_time - self.job.arrival_time

    @property
    def flow_time(self) -> int:
        return self.completion_time - self.job.arrival_time

    @property
    def tardiness(self) -> int:
        return max(0, self.completion_time - self.job.deadline)
