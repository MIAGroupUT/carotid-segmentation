import pytest
from click.testing import CliRunner

from carotid.cli import cli


# Test for the first level at the command line
@pytest.fixture(
    params=[
        "transform",
        "compare",
    ]
)
def cli_args_first_lv(request):
    task = request.param
    return task


def test_first_lv(cli_args_first_lv):
    runner = CliRunner()
    task = cli_args_first_lv
    print(f"Testing input cli {task}")
    result = runner.invoke(cli, f"{task} -h")
    assert result.exit_code == 0


@pytest.fixture(
    params=[
        "centerline",
        "heatmap",
        "polar",
        "contour",
        "segmentation",
        "pipeline",
    ]
)
def cli_args_transform_lv(request):
    task = request.param
    return task


def test_transform_lv(cli_args_transform_lv):
    runner = CliRunner()
    task = cli_args_transform_lv
    print(f"Testing input cli {task}")
    result = runner.invoke(cli, f"transform {task} -h")
    assert result.exit_code == 0


@pytest.fixture(
    params=[
        "centerline",
    ]
)
def cli_args_compare_lv(request):
    task = request.param
    return task


def test_compare_lv(cli_args_compare_lv):
    runner = CliRunner()
    task = cli_args_compare_lv
    print(f"Testing input cli {task}")
    result = runner.invoke(cli, f"compare {task} -h")
    assert result.exit_code == 0
