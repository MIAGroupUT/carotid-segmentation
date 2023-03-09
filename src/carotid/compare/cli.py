import click
from carotid.utils.cli_param.decorators import OrderedGroup
from carotid.compare.centerline.cli import cli as centerline_cli
from carotid.compare.contour.cli import cli as contour_cli


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


cli.add_command(centerline_cli)
cli.add_command(contour_cli)
