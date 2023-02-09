import click
from carotid.utils import cli_param


@click.command(
    "centerline_transform",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@cli_param.argument.output_dir
@click.option(
    "--heatmap_dir",
    "-hdir",
    type=click.Path(exists=True),
    default=None,
    help="Path to the output directory of heatmap_transform, if different from output_dir.",
)
@cli_param.option.config_path
@cli_param.option.participant
def cli(
    output_dir,
    heatmap_dir,
    config_path,
    participant,
) -> None:
    """
    Extract centerlines from heatmaps with the Dijkstra algorithm.

    OUTPUT_DIR is the path to the directory containing the results.
    """
    from .pipeline import apply_transform

    apply_transform(
        output_dir=output_dir,
        heatmap_dir=heatmap_dir,
        config_path=config_path,
        participant_list=participant,
    )


if __name__ == "__main__":
    cli()
