"""Tabular Q-Learning over four conventional scheduling policies."""

import json
import random
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Sequence

from adaptivequeue.metrics import Metrics, calculate_metrics
from adaptivequeue.models import Job, JobResult
from adaptivequeue.policies import edf, fifo, priority, spt
from adaptivequeue.simulation import Simulation, simulate
from adaptivequeue.workloads import WORKLOADS, WorkloadConfig

State = tuple[int, int, int, bool]
ACTION_NAMES = ("fifo", "spt", "edf", "priority")
ACTION_POLICIES = (fifo, spt, edf, priority)
STATE_ENCODING = {
    "fields": [
        "queue_length_bucket",
        "min_slack_bucket",
        "short_proportion_bucket",
        "high_priority",
    ],
    "queue_length_upper_bounds": [1, 3, 7],
    "slack": "deadline - current_time - duration",
    "slack_upper_bounds": [-1, 2, 6],
    "short_duration_max": 2,
    "short_proportion_boundaries": ["1/3", "2/3"],
    "high_priority_min": 4,
}
REWARD = "-priority_weight * (tardiness + 0.1 * flow_time)"


def encode_state(waiting_jobs: Sequence[Job], current_time: int) -> State:
    if not waiting_jobs:
        raise ValueError("state encoding requires a nonempty ready queue")
    size = len(waiting_jobs)
    queue_bucket = sum(size > bound for bound in (1, 3, 7))
    slack = min(job.deadline - current_time - job.duration for job in waiting_jobs)
    slack_bucket = sum(slack > bound for bound in (-1, 2, 6))
    short_count = sum(job.duration <= 2 for job in waiting_jobs)
    short_bucket = int(3 * short_count >= size) + int(3 * short_count >= 2 * size)
    return (
        queue_bucket,
        slack_bucket,
        short_bucket,
        any(job.priority_weight >= 4 for job in waiting_jobs),
    )


@dataclass(frozen=True)
class Transition:
    job: JobResult
    reward: float
    next_state: State | None
    done: bool


def step_action(simulation: Simulation, action: int) -> Transition:
    if not 0 <= action < len(ACTION_POLICIES):
        raise ValueError("action must be an index from 0 to 3")
    record = simulation.step(ACTION_POLICIES[action])
    next_state = (
        None if simulation.done else encode_state(simulation.waiting_jobs, simulation.current_time)
    )
    return Transition(record, -record.cost, next_state, simulation.done)


@dataclass(frozen=True)
class LearningConfig:
    learning_rate: float = 0.1
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("learning_rate must be in (0, 1] and gamma in [0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")

    def epsilon(self, episode: int) -> float:
        return max(self.epsilon_end, self.epsilon_start * self.epsilon_decay**episode)


@dataclass(frozen=True)
class TrainingInfo:
    workload: str
    workload_config: WorkloadConfig
    episodes: int
    seed: int


class QLearningAgent:
    def __init__(self, *, seed: int, config: LearningConfig = LearningConfig()) -> None:
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        self.seed = seed
        self.config = config
        self.training: TrainingInfo | None = None
        self.q_table: dict[State, list[float]] = {}
        self._rng = random.Random(seed)

    def greedy_action(self, state: State) -> int:
        values = self.q_table.get(state, [0.0] * len(ACTION_NAMES))
        return max(range(len(ACTION_NAMES)), key=lambda action: values[action])

    def choose_action(self, state: State, *, epsilon: float) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon > 0 and self._rng.random() < epsilon:
            return self._rng.randrange(len(ACTION_NAMES))
        return self.greedy_action(state)

    def update(self, state: State, action: int, reward: float, next_state: State | None) -> None:
        if not 0 <= action < len(ACTION_NAMES):
            raise ValueError("action must be an index from 0 to 3")
        target = reward
        if next_state is not None:
            target += self.config.gamma * max(
                self.q_table.get(next_state, [0.0] * len(ACTION_NAMES))
            )
        values = self.q_table.setdefault(state, [0.0] * len(ACTION_NAMES))
        values[action] += self.config.learning_rate * (target - values[action])

    def policy(self, waiting_jobs: Sequence[Job], current_time: int) -> Job:
        """A frozen, greedy policy compatible with simulate()."""
        action = self.greedy_action(encode_state(waiting_jobs, current_time))
        return ACTION_POLICIES[action](waiting_jobs, current_time)

    def save(self, path: str | Path) -> None:
        data = {
            "version": 1,
            "actions": ACTION_NAMES,
            "state_encoding": STATE_ENCODING,
            "reward": REWARD,
            "seed": self.seed,
            "learning_config": asdict(self.config),
            "training": asdict(self.training) if self.training else None,
            "q_table": [
                {"state": state, "values": values} for state, values in sorted(self.q_table.items())
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            data["version"] != 1
            or data["actions"] != list(ACTION_NAMES)
            or data["state_encoding"] != STATE_ENCODING
            or data["reward"] != REWARD
        ):
            raise ValueError("incompatible Q-table format or scheduling formulation")
        agent = cls(seed=data["seed"], config=LearningConfig(**data["learning_config"]))
        if data["training"] is not None:
            info = data["training"]
            workload = info["workload_config"]
            workload["duration_range"] = tuple(workload["duration_range"])
            agent.training = TrainingInfo(
                info["workload"], WorkloadConfig(**workload), info["episodes"], info["seed"]
            )
        for row in data["q_table"]:
            raw = row["state"]
            if len(raw) != 4 or len(row["values"]) != len(ACTION_NAMES):
                raise ValueError("invalid Q-table row dimensions")
            state: State = (raw[0], raw[1], raw[2], raw[3])
            if (
                any(type(value) is not int for value in state[:3])
                or not (0 <= state[0] < 4 and 0 <= state[1] < 4 and 0 <= state[2] < 3)
                or not isinstance(state[3], bool)
            ):
                raise ValueError("invalid Q-table state")
            values = [float(value) for value in row["values"]]
            if not all(isfinite(value) for value in values):
                raise ValueError("Q-values must be finite")
            agent.q_table[state] = values
        return agent


def train(
    *,
    workload: str = "mixed",
    episodes: int = 1000,
    seed: int = 42,
    workload_config: WorkloadConfig = WorkloadConfig(),
    learning_config: LearningConfig = LearningConfig(),
) -> QLearningAgent:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    agent = QLearningAgent(seed=seed, config=learning_config)
    agent.training = TrainingInfo(workload, workload_config, episodes, seed)
    for episode in range(episodes):
        simulation = Simulation(WORKLOADS[workload](seed + episode, workload_config))
        state = (
            None
            if simulation.done
            else encode_state(simulation.waiting_jobs, simulation.current_time)
        )
        epsilon = learning_config.epsilon(episode)
        while state is not None:
            action = agent.choose_action(state, epsilon=epsilon)
            transition = step_action(simulation, action)
            agent.update(state, action, transition.reward, transition.next_state)
            state = transition.next_state
    return agent


def evaluate(
    agent: QLearningAgent,
    *,
    workload: str = "mixed",
    episodes: int = 20,
    seed: int | None = None,
    workload_config: WorkloadConfig = WorkloadConfig(),
) -> Metrics:
    """Pool job metrics across independently seeded, greedy evaluation episodes."""
    training = agent.training
    if seed is None:
        seed = training.seed + training.episodes if training else agent.seed + 1
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed nonnegative")
    if training and seed < training.seed + training.episodes and seed + episodes > training.seed:
        raise ValueError("evaluation workload seeds must not overlap the training seeds")
    records: list[JobResult] = []
    for episode in range(episodes):
        jobs = WORKLOADS[workload](seed + episode, workload_config)
        records.extend(simulate(jobs, agent.policy).jobs)
    return calculate_metrics(records)
