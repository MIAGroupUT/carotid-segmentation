from os import path
from carotid.train.contour.pipeline import train
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "contour", "input")

    train(
        output_dir=tmp_dir,
        raw_dir=path.join(test_dir, "..", "raw_dir"),
        contour_dir=input_dir,
        contour_tsv=path.join(input_dir, "contours.tsv"),
        train_config_path=path.join(test_dir, "contour", "test_args_train.toml"),
        force=True
    )

    shutil.rmtree(tmp_dir)
