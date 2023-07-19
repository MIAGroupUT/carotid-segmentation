import click


config_path = click.option(
    "--config_path",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to a TOML file to set transforms parameters.",
)
train_config_path = click.option(
    "--train_config_path",
    "-tc",
    type=click.Path(exists=True),
    default=None,
    help="Path to a TOML file to set training parameters.",
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
    type=click.Choice(["cpu", "cuda", "mps"]),
    default=None,
    help="Device used for deep learning computations.",
)
force = click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force to run the transform even if it was already performed in output_dir.",
)
verbose = click.option(
    "-v",
    "--verbose",
    count=True,
    help="Each occurrence increases the level of verbosity. Default only outputs warnings and errors.",
)
