import click
from carotid.transforms.heatmap.cli import cli as heatmap_cli
from carotid.transforms.centerline.cli import cli as centerline_cli
from carotid.transforms.polar.cli import cli as polar_cli
from carotid.transforms.contour.cli import cli as contour_cli
from carotid.transforms.segmentation.cli import cli as segmentation_cli
from carotid.transforms.pipeline.cli import cli as pipeline_cli
from carotid.utils.cli_param.decorators import OrderedGroup


CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True, cls=OrderedGroup, name="transform")
def cli():
    """Transform 3D black-blood MRI in different steps of the carotid segmentation algorithm."""
    pass


cli.add_command(heatmap_cli)
cli.add_command(centerline_cli)
cli.add_command(polar_cli)
cli.add_command(contour_cli)
cli.add_command(segmentation_cli)
cli.add_command(pipeline_cli)
