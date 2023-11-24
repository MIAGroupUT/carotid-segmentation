from os import path, remove
from carotid.train.contour.pipeline import train
import shutil
import pytest

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


@pytest.fixture(
    params=[
        "train2d",
        "train3d",
    ]
)
def pipeline_args(request):
    task = request.param
    return task


def test_pipeline(pipeline_args):
    args_filename = f"test_args_{pipeline_args}.toml"
    tmp_dir = path.join(test_dir, f"tmp{pipeline_args}")
    input_dir = path.join(test_dir, "contour", "input")
    raw_dir = path.join(test_dir, "..", "raw_dir")

    # Duplicate participant to pretend there are two
    if not path.exists(path.join(raw_dir, "0_P126_U.mha")):
        shutil.copy(
            path.join(raw_dir, "0_P125_U.mha"), path.join(raw_dir, "0_P126_U.mha")
        )
    if not path.exists(path.join(input_dir, "0_P126_U")):
        shutil.copytree(
            path.join(input_dir, "0_P125_U"), path.join(input_dir, "0_P126_U")
        )

    train(
        output_dir=tmp_dir,
        raw_dir=raw_dir,
        contour_dir=input_dir,
        contour_tsv=path.join(input_dir, "contours.tsv"),
        train_config_path=path.join(test_dir, "contour", args_filename),
        force=True,
    )

    remove(path.join(raw_dir, "0_P126_U.mha"))
    shutil.rmtree(path.join(input_dir, "0_P126_U"))
    shutil.rmtree(tmp_dir)
