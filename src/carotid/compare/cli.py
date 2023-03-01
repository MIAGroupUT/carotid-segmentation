import click
from carotid.utils.cli_param.decorators import OrderedGroup


CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True, cls=OrderedGroup, name="compare")
def cli():
    """Compare the outputs of two different settings of the same transform."""
    pass


if __name__ == "__main__":
    cli()
