import click


@click.command(
    "segmentation_transform",
    no_args_is_help=True,
    context_settings={"show_default": True},
)
@click.argument(
    "output_dir",
    type=click.Path(writable=True),
)
@click.option(
    "--polar_dir",
    "-pdir",
    type=click.Path(exists=True),
    default=None,
    help="Path to the output directory of polar_transform, if different from output_dir.",
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
    output_dir,
    polar_dir,
    config_path,
    participant,
    device,
) -> None:
    """
    Extracting contours from raw images, polar_images and centerlines found with previous steps.

    OUTPUT_DIR is the path to the directory containing the results.
    """
    from .pipeline import apply_transform

    apply_transform(
        output_dir=output_dir,
        polar_dir=polar_dir,
        config_path=config_path,
        participant_list=participant,
        device=device,
    )


if __name__ == "__main__":
    cli()
