import subprocess
import sys
from pathlib import Path


def test_module_cli_displays_metrics() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "adaptivequeue",
            "--workload",
            "mixed",
            "--policy",
            "edf",
            "--seed",
            "7",
            "--jobs",
            "12",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Workload: mixed | Policy: edf | Seed: 7 | Jobs: 12" in result.stdout
    for label in (
        "Primary cost J",
        "Deadline miss rate",
        "Average waiting time",
        "95th-percentile waiting time",
        "Average flow time",
    ):
        assert label in result.stdout


def test_invalid_cli_configuration_has_a_useful_error() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "adaptivequeue", "--arrival-rate", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "arrival_rate must be finite and positive" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_training_and_evaluation(tmp_path: Path) -> None:
    model = tmp_path / "policy.json"
    training = subprocess.run(
        [
            sys.executable,
            "-m",
            "adaptivequeue",
            "train",
            "--workload",
            "bursty",
            "--episodes",
            "4",
            "--jobs",
            "8",
            "--seed",
            "7",
            "--learning-rate",
            "0.3",
            "--output",
            str(model),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Trained 4 episodes" in training.stdout
    assert model.is_file()
    evaluation = subprocess.run(
        [
            sys.executable,
            "-m",
            "adaptivequeue",
            "evaluate",
            "--workload",
            "bursty",
            "--episodes",
            "2",
            "--jobs",
            "8",
            "--model",
            str(model),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Policy: qlearning | Episodes: 2" in evaluation.stdout
    assert "Primary cost J" in evaluation.stdout
    assert "95th-percentile waiting time" in evaluation.stdout
