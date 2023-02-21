import click


output_dir = click.argument(
    "output_dir",
    type=click.Path(writable=True),
)
raw_dir = click.argument(
    "raw_dir",
    type=click.Path(exists=True, readable=True),
)
heatmap_model_dir = click.argument(
    "heatmap_model_dir",
    type=click.Path(exists=True),
)
contour_model_dir = click.argument(
    "contour_model_dir",
    type=click.Path(exists=True),
)
