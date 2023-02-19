import click


config_path = click.option(
    "--config_path",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to a TOML file to set parameters.",
)
participant = click.option(
    "--participant",
    "-p",
    type=str,
    default=None,
    multiple=True,
    help="Names of the participants who will be used.",
)
device = click.option(
    "--device",
    "-d",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help="Device used for deep learning computations.",
)
