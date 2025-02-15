import pathlib

import click

from fmp.datasets import fingerspelling5


@click.command()
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=True, file_okay=False
    ),
)
def main(dataset_dir: pathlib.Path):
    filter_nans = True
    dataset_name = dataset_dir.name
    data_df = fingerspelling5.utils.read_csv(
        dataset_dir / f"{dataset_name}.csv", filter_nans=filter_nans
    )
    image_file_df = data_df.loc[:, ["letter", "person", "img_file"]].reset_index(
        drop=True
    )
    image_file_df.to_csv(dataset_dir / "image_files.csv", index=False)


if __name__ == "__main__":
    main()
