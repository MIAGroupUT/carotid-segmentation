import click


@click.command(
    "centerline_transform", no_args_is_help=True, context_settings={"show_default": True}
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
    "config_path",
    type=click.Path(exists=True),
)
@click.argument(
    "output_dir",
    type=click.Path(exists=True, writable=True),
)
@click.option(
    "--participant", "-p", type=str, default=None, multiple=True
)
@click.option(
    "--device", "-d", type=click.Choice(["cpu", "cuda"]), default="cuda"
)
def cli(
    raw_dir,
    model_dir,
    config_path,
    output_dir,
    participant,
    device,
) -> None:
    """
    Extracting centerlines from raw images using pre-trained U-Nets.

    RAW_DIR is the path to raw data folder.

    MODEL_DIR is the path to a directory where the models are stored.

    CONFIG_PATH is the path to a TOML file containing the parameters to run the pipeline.

    OUTPUT_DIR is the path to the directory containing the results.
    """
    from pipeline import apply_transform

    apply_transform(
        raw_dir, model_dir, config_path, output_dir, participant, device=device
    )


if __name__ == "__main__":
    cli()
