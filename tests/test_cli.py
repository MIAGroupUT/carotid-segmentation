import pytest
from click.testing import CliRunner

from carotid.cli import cli


# Test for the first level at the command line
@pytest.fixture(
    params=[
        "convert",
        "transform",
        "compare",
        "train",
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
        "contour",
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


@pytest.fixture(
    params=[
        "contour",
    ]
)
def cli_args_train_lv(request):
    task = request.param
    return task


def test_train_lv(cli_args_train_lv):
    runner = CliRunner()
    task = cli_args_transform_lv
    print(f"Testing input cli {task}")
    result = runner.invoke(cli, f"train {task} -h")
    assert result.exit_code == 0


@pytest.fixture(
    params=[
        "miccai2020",
        "miccai2022",
    ]
)
def cli_args_convert_lv(request):
    task = request.param
    return task


def test_convert_lv(cli_args_convert_lv):
    runner = CliRunner()
    task = cli_args_transform_lv
    print(f"Testing input cli {task}")
    result = runner.invoke(cli, f"convert {task} -h")
    assert result.exit_code == 0
