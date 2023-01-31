import click


@click.command(
    "heatmap_transform",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@click.argument(
    "raw_dir",
    type=click.Path(exists=True),
)
@click.argument(
    "model_dir",
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
    model_dir,
    output_dir,
    config_path,
    participant,
    device,
) -> None:
    """
    Extracting centerlines from raw images using pre-trained U-Nets.

    RAW_DIR is the path to raw data folder.

    MODEL_DIR is the path to a directory where the models are stored.

    OUTPUT_DIR is the path to the directory containing the results.
    """
    from .pipeline import apply_transform

    apply_transform(
        raw_dir, model_dir, output_dir, config_path, participant, device=device
    )


if __name__ == "__main__":
    cli()
