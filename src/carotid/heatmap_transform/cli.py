import click
from carotid.utils import cli_param


@click.command(
    "heatmap_transform",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@cli_param.argument.raw_dir
@cli_param.argument.heatmap_model_dir
@cli_param.argument.output_dir
@cli_param.option.config_path
@cli_param.option.participant
@cli_param.option.device
@cli_param.option.force
def cli(
    raw_dir,
    heatmap_model_dir,
    output_dir,
    config_path,
    participant,
    device,
    force,
) -> None:
    """
    Extract heatmaps from raw images using pre-trained U-Nets.

    RAW_DIR is the path to raw data folder.

    HEATMAP_MODEL_DIR is the path to a directory where the models are stored.

    OUTPUT_DIR is the path to the directory containing the results.
    """
    from .pipeline import apply_transform

    apply_transform(
        raw_dir=raw_dir,
        model_dir=heatmap_model_dir,
        output_dir=output_dir,
        config_path=config_path,
        participant_list=participant,
        device=device,
        force=force,
    )


if __name__ == "__main__":
    cli()
