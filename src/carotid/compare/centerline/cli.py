import click
from carotid.utils import cli_param


@click.command(
    name="centerline",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@click.argument(
    "transform1_dir",
    type=click.Path(exists=True)
)
@click.argument(
    "transform2_dir",
    type=click.Path(exists=True)
)
@click.argument(
    "output_path",
    type=click.Path(writable=True)
)
def cli(
    transform1_dir,
    transform2_dir,
    output_path,
) -> None:
    """
    Compare the centerlines in two output directories of `carotid transform centerline`.

    TRANSFORM1_DIR is the path to the first directory to compare.

    TRANSFORM2_DIR is the path to the second directory to compare.

    OUTPUT_PATH is the path to the output TSV file.
    """
    from .pipeline import compare

    compare(
        transform1_dir=transform1_dir,
        transform2_dir=transform2_dir,
        output_path=output_path,
    )
