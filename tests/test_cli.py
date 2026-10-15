import subprocess
import sys


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
