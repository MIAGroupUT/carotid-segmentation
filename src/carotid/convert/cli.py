import click
from carotid.utils.cli_param.decorators import OrderedGroup
from carotid.convert.miccai2020.cli import cli as miccai2020cli


CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


@click.group(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    cls=OrderedGroup,
    name="convert",
)
def cli():
    """Convert raw data sets to the format used by the library."""
    pass


cli.add_command(miccai2020cli)
