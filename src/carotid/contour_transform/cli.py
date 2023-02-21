import click
from carotid.utils import cli_param


@click.command(
    "contour_transform",
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
def cli(
    output_dir,
    contour_model_dir,
    polar_dir,
    config_path,
    participant,
    device,
) -> None:
    """
    Extract contours from raw images, polar_images and centerlines found with previous steps.

    OUTPUT_DIR is the path to the directory containing the results.
    """
    from .pipeline import apply_transform

    apply_transform(
        output_dir=output_dir,
        model_dir=contour_model_dir,
        polar_dir=polar_dir,
        config_path=config_path,
        participant_list=participant,
        device=device,
    )


if __name__ == "__main__":
    cli()
