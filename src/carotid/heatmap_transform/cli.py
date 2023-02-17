import click
from carotid.utils import cli_param


@click.command(
    "heatmap_transform",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@cli_param.argument.raw_dir
@click.argument(
    "model_dir",
    type=click.Path(exists=True),
)
@cli_param.argument.output_dir
@cli_param.option.config_path
@cli_param.option.participant
@cli_param.option.device
def cli(
    raw_dir,
    model_dir,
    output_dir,
    config_path,
    participant,
    device,
) -> None:
    """
    Extract heatmaps from raw images using pre-trained U-Nets.

    RAW_DIR is the path to raw data folder.

    MODEL_DIR is the path to a directory where the models are stored.

    OUTPUT_DIR is the path to the directory containing the results.
    """
    from .pipeline import apply_transform

    apply_transform(
        raw_dir=raw_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        config_path=config_path,
        participant_list=participant,
        device=device,
    )


if __name__ == "__main__":
    cli()
