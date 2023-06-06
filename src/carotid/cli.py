import click
from carotid.utils.cli_param.decorators import OrderedGroup
from carotid.transform.cli import cli as transform_cli
from carotid.compare.cli import cli as compare_cli
from carotid.train.cli import cli as train_cli


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


cli.add_command(transform_cli)
cli.add_command(compare_cli)
cli.add_command(train_cli)

if __name__ == "__main__":
    cli()
