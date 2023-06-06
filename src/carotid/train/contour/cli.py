import click
from carotid.utils import cli_param


@click.command(
    "contour",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@cli_param.argument.output_dir
@cli_param.argument.raw_dir
@click.argument(
    "contour_dir",
    type=click.Path(exists=True)
)
@click.option(
    "--polar_dir",
    "-pdir",
    type=click.Path(exists=True),
    default=None,
    help="Path to the output directory of polar_transform, if different from output_dir.",
)
@click.option(
    "--contour_tsv", "-ctsv",
    type=click.Path(exists=True),
    default=None,
    help="Path to a TSV file specifying which contours should be used in the training and validation sets."
)
@cli_param.option.train_config_path
@cli_param.option.config_path
@cli_param.option.device
@cli_param.option.force
def cli(
    output_dir,
    raw_dir,
    contour_dir,
    contour_tsv,
    train_config_path,
    config_path,
    device,
    force,
) -> None:
    """
    Train a network from a set of contours.

    OUTPUT_DIR is the path to the directory containing the results.

    CONTOUR_MODEL_DIR is the path to a directory where the models for contour extraction are stored.
    """
    from .pipeline import train

    train(
        output_dir=output_dir,
        raw_dir=raw_dir,
        contour_dir=contour_dir,
        contour_tsv=contour_tsv,
        train_config_path=train_config_path,
        polar_config_dict=config_path,
        device=device,
        force=force,
    )
