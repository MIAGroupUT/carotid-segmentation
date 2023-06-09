import click
from carotid.utils import cli_param


@click.command(
    "contour",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@cli_param.argument.output_dir
@cli_param.argument.contour_model_dir
@click.option(
    "--polar_dir",
    "-pdir",
    type=click.Path(exists=True),
    default=None,
    help="Path to the output directory of polar_transform, if different from output_dir.",
)
@cli_param.option.config_path
@cli_param.option.participant
@cli_param.option.device
@cli_param.option.force
def cli(
    output_dir,
    contour_model_dir,
    polar_dir,
    config_path,
    participant,
    device,
    force,
) -> None:
    """
    Extract contours from raw images and corresponding polar images computed by polar_transform.

    OUTPUT_DIR is the path to the directory containing the results.

    CONTOUR_MODEL_DIR is the path to a directory where the models for contour extraction are stored.
    """
    from .pipeline import apply_transform

    apply_transform(
        output_dir=output_dir,
        model_dir=contour_model_dir,
        polar_dir=polar_dir,
        config_path=config_path,
        participant_list=participant,
        device=device,
        force=force,
    )
