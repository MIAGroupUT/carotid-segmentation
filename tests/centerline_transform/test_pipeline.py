import pytest
from os import path
from carotid.centerline_transform.pipeline import apply_transform
import shutil

test_dir = path.dirname(path.realpath(__file__))

# Test for the first level at the command line
@pytest.fixture(
    params=[
        path.join(test_dir, "config1.toml"),
        path.join(test_dir, "config2.toml"),
    ]
)
def config_path(request):
    task = request.param
    return task


def test_first_lv(config_path):

    apply_transform(
        raw_dir=path.join(test_dir, "../raw_dir"),
        model_dir=path.join(test_dir, "../models/centerline_transform"),
        config_path=config_path,
        output_dir=path.join(test_dir, "output_dir"),
        device="cpu",
        debug=True,
    )

    shutil.rmtree(path.join(test_dir, "output_dir"))
