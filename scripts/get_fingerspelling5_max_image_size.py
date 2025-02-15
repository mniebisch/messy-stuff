import pathlib

import click
import pandas as pd


@click.command()
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=True, file_okay=False
    ),
)
def main(dataset_dir: pathlib.Path):
    sizes_df = pd.read_csv(dataset_dir / "image_sizes.csv")

    max_width = sizes_df["width"].max()
    max_height = sizes_df["height"].max()

    click.echo(f"Max width: {max_width}, Max height: {max_height}")


if __name__ == "__main__":
    main()
