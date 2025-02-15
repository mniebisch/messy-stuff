import pathlib

import click
import pandas as pd
import tqdm
from PIL import Image


@click.command()
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=True, file_okay=False
    ),
)
@click.option(
    "--images-dir",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=True, file_okay=False
    ),
)
def main(dataset_dir: pathlib.Path, images_dir: pathlib.Path):
    data_df = pd.read_csv(dataset_dir / "image_files.csv")

    img_file_list = data_df["img_file"].tolist()

    image_data = []
    for img_file in tqdm.tqdm(img_file_list):
        try:
            with Image.open(images_dir / img_file) as img:
                width, height = img.size
                image_data.append(
                    {"img_file": img_file, "width": width, "height": height}
                )
        except Exception as e:
            print(f"Error reading {img_file}: {e}")

    image_sizes_df = pd.DataFrame(image_data)
    image_sizes_df.to_csv(dataset_dir / "image_sizes.csv", index=False)


if __name__ == "__main__":
    main()
