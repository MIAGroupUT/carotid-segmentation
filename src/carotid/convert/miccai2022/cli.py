import click
from carotid.utils import cli_param


@click.command(
    "miccai2022",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@cli_param.argument.original_dir
@cli_param.argument.out_raw_dir
@cli_param.argument.annotation_dir
def cli(
    original_dir,
    raw_dir,
    annotation_dir,
) -> None:
    """
    Converts the data set of the MICCAI grand challenge 2022.
    https://zenodo.org/record/6481870

    ORIGINAL_DIR is the path to the data as downloaded from the grand challenge website.

    RAW_DIR is the path to the directory containing the formatted raw data.

    ANNOTATION_DIR is the path to the directory containing the formatted annotations.
    """
    from .pipeline import convert

    convert(
        original_dir=original_dir,
        raw_dir=raw_dir,
        annotation_dir=annotation_dir,
    )
