import click
from carotid.utils import cli_param


@click.command(
    name="polar",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@cli_param.argument.output_dir
@click.option(
    "--centerline_dir",
    "-cdir",
    type=click.Path(exists=True),
    default=None,
    help="Path to the output directory of centerline_transform, if different from output_dir.",
)
@cli_param.option.config_path
@cli_param.option.participant
@cli_param.option.force
def cli(
    output_dir,
    centerline_dir,
    config_path,
    participant,
    force,
) -> None:
    """
    Extract polar images from raw images based on the centerlines found with centerline_transform.

    OUTPUT_DIR is the path to the directory containing the results.
    """
    from .pipeline import apply_transform

    apply_transform(
        output_dir=output_dir,
        centerline_dir=centerline_dir,
        config_path=config_path,
        participant_list=participant,
        force=force,
    )
