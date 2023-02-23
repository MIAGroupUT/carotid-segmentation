import click
from .heatmap_transform.cli import cli as heatmap_cli
from .centerline_transform.cli import cli as centerline_cli
from .polar_transform.cli import cli as polar_cli
from .contour_transform.cli import cli as contour_cli
from .segmentation_transform.cli import cli as segmentation_cli
from .pipeline_transform.cli import cli as pipeline_cli
from .utils.cli_param.decorators import OrderedGroup


CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True, cls=OrderedGroup)
@click.version_option()
def cli():
    """carotid-segmentation command line."""
    pass


cli.add_command(heatmap_cli)
cli.add_command(centerline_cli)
cli.add_command(polar_cli)
cli.add_command(contour_cli)
cli.add_command(segmentation_cli)
cli.add_command(pipeline_cli)

if __name__ == "__main__":
    cli()
