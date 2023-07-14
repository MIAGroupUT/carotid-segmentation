import click
from carotid.utils import cli_param


@click.command(
    "miccai2020",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@cli_param.argument.original_dir
@cli_param.argument.output_dir
def cli(
    original_dir,
    output_dir,
) -> None:
    """
    Converts the data set of the MICCAI grand challenge 2020.
    https://vessel-wall-segmentation.grand-challenge.org/

    ORIGINAL_DIR is the path to the data as downloaded from the grand challenge website.

    OUTPUT_DIR is the path to the directory containing the formatted data set.
    """
    from .pipeline import convert

    convert(
        original_dir=original_dir,
        output_dir=output_dir,
    )
