import click


@click.command(
    "pipeline_transform",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@click.argument(
    "raw_dir",
    type=click.Path(exists=True),
)
@click.argument(
    "heatmap_model_dir",
    type=click.Path(exists=True),
)
@click.argument(
    "segmentation_model_dir",
    type=click.Path(exists=True),
)
@click.argument(
    "output_dir",
    type=click.Path(writable=True),
)
@click.option(
    "--config_path",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to a TOML file to set parameters.",
)
@click.option("--participant", "-p", type=str, default=None, multiple=True)
@click.option("--device", "-d", type=click.Choice(["cpu", "cuda"]), default="cuda")
def cli(
    raw_dir,
    heatmap_model_dir,
    segmentation_model_dir,
    output_dir,
    config_path,
    participant,
    device,
) -> None:
    """
    Extracting centerlines from heatmaps with the Dijkstra algorithm.

    OUTPUT_DIR is the path to the directory containing the results.
    """
    from .pipeline import apply_transform

    apply_transform(
        raw_dir=raw_dir,
        output_dir=output_dir,
        heatmap_model_dir=heatmap_model_dir,
        segmentation_model_dir=segmentation_model_dir,
        config_path=config_path,
        participant_list=participant,
        device=device,
    )


if __name__ == "__main__":
    cli()
