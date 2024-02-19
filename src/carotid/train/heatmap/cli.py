import click
from carotid.utils import cli_param


@click.command(
    "heatmap",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@cli_param.argument.output_dir
@cli_param.argument.raw_dir
@cli_param.argument.annotation_dir
@click.option(
    "--centerline_tsv",
    "-ctsv",
    type=click.Path(exists=True),
    default=None,
    help="Path to a TSV file specifying which centerlines should be used in the training and validation sets.",
)
@cli_param.option.train_config_path
@cli_param.option.config_path
@cli_param.option.device
@cli_param.option.force
def cli(
    output_dir,
    raw_dir,
    annotation_dir,
    centerline_tsv,
    train_config_path,
    config_path,
    device,
    force,
) -> None:
    """
    Train a network from a set of centerlines.

    OUTPUT_DIR is the path to the directory containing the results.

    RAW_DIR is the path to raw data folder.

    ANNOTATION_DIR is the path to the data folder containing centerline annotations.
    """
    from .pipeline import train

    train(
        output_dir=output_dir,
        raw_dir=raw_dir,
        centerline_dir=annotation_dir,
        centerline_tsv=centerline_tsv,
        train_config_path=train_config_path,
        polar_config_dict=config_path,
        device=device,
        force=force,
    )
