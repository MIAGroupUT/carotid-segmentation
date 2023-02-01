import pytest
from click.testing import CliRunner

from carotid.cli import cli


# Test for the first level at the command line
@pytest.fixture(params=["centerline_transform", "heatmap_transform", "polar_transform"])
def cli_args_first_lv(request):
    task = request.param
    return task


def test_first_lv(cli_args_first_lv):
    runner = CliRunner()
    task = cli_args_first_lv
    print(f"Testing input cli {task}")
    result = runner.invoke(cli, f"{task} -h")
    assert result.exit_code == 0
