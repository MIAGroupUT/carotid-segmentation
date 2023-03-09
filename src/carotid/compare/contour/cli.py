import click
from carotid.utils import cli_param


@click.command(
    name="contour",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@click.argument(
    "transform_dir",
    type=click.Path(exists=True)
)
@click.argument(
    "reference_dir",
    type=click.Path(exists=True)
)
@click.argument(
    "output_dir",
    type=click.Path(writable=True, exists=True)
)
def cli(
    transform_dir,
    reference_dir,
    output_dir,
) -> None:
    """
    Compare the contours in two output directories of `carotid transform contour`.

    TRANSFORM1_DIR is the path to the first directory to compare.

    REFERENCE_DIR is the path to the reference.

    OUTPUT_DIR is the path to the directory where the TSV files are written.
    """
    from .pipeline import compare

    compare(
        transform_dir=transform_dir,
        reference_dir=reference_dir,
        output_dir=output_dir,
    )
