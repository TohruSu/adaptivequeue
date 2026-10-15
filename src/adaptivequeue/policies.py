"""Policies receive only the jobs currently waiting for the worker."""

import random
from typing import Callable, Sequence

from adaptivequeue.models import Job

Policy = Callable[[Sequence[Job]], Job]


def fifo(waiting: Sequence[Job]) -> Job:
    return min(waiting, key=lambda job: (job.arrival_time, job.job_id))


def spt(waiting: Sequence[Job]) -> Job:
    return min(waiting, key=lambda job: (job.duration, job.arrival_time, job.job_id))


def edf(waiting: Sequence[Job]) -> Job:
    return min(waiting, key=lambda job: (job.deadline, job.arrival_time, job.job_id))


def priority(waiting: Sequence[Job]) -> Job:
    return min(waiting, key=lambda job: (-job.priority_weight, job.arrival_time, job.job_id))


POLICIES: dict[str, Policy] = {"fifo": fifo, "spt": spt, "edf": edf, "priority": priority}
POLICY_NAMES = (*POLICIES, "random")


def make_policy(name: str, *, seed: int) -> Policy:
    """Create a policy; recreate it with the same seed to replay a random run."""
    if name == "random":
        rng = random.Random(seed)
        return rng.choice
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy: {name}") from None
