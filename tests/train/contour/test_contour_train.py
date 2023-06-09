from os import path, remove
from carotid.train.contour.pipeline import train
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "contour", "input")
    raw_dir = path.join(test_dir, "..", "raw_dir")

    # Duplicate participant to pretend there are two
    shutil.copy(path.join(raw_dir, "0_P125_U.mha"), path.join(raw_dir, "0_P126_U.mha"))
    shutil.copytree(path.join(input_dir, "0_P125_U"), path.join(input_dir, "0_P126_U"))

    train(
        output_dir=tmp_dir,
        raw_dir=raw_dir,
        contour_dir=input_dir,
        contour_tsv=path.join(input_dir, "contours.tsv"),
        train_config_path=path.join(test_dir, "contour", "test_args_train.toml"),
        force=True,
    )

    remove(path.join(raw_dir, "0_P126_U.mha"))
    shutil.rmtree(path.join(input_dir, "0_P126_U"))
    shutil.rmtree(tmp_dir)
