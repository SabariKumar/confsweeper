"""
Pytest configuration for GPU integration tests.

Tests marked @pytest.mark.gpu are skipped by default. To run them:

    pixi run python -m pytest tests/integration/ --gpu

All gpu tests are also skipped automatically if no CUDA device is present,
regardless of the --gpu flag.
"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run GPU integration tests (requires a CUDA device and nvmolkit).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: requires a CUDA GPU and nvmolkit — run with --gpu",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip = pytest.mark.skip(reason="pass --gpu to run GPU integration tests")
    elif not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA device detected")
    else:
        return

    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
