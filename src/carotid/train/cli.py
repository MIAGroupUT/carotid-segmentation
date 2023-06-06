import click
from carotid.train.contour.cli import cli as contour_cli
from carotid.utils.cli_param.decorators import OrderedGroup


CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True, cls=OrderedGroup, name="train")
def cli():
    """Train network to perform the tasks used by the pipeline."""
    pass


cli.add_command(contour_cli)
