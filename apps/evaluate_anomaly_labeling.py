import pathlib

import click
import pandas as pd
import plotly.express as px


@click.command()
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=True, file_okay=False
    ),
)
@click.option("--dataset-name", required=True, type=str)
def main(dataset_dir: pathlib.Path, dataset_name: str) -> None:
    """
    python apps/evaluate_anomaly_labeling.py \ 
        --dataset-dir data/fingerspelling5 \
        --dataset-name fingerspelling5_singlehands_sorted
    
    """

    label_file = dataset_dir / dataset_name / f"{dataset_name}__data_quality.csv"

    df = pd.read_csv(label_file)

    overview = df.groupby(["person", "letter"])[["views"]].agg(lambda x: any(x == 0))
    overview = overview.reset_index()

    overview_matrix = overview.pivot(index="person", columns="letter", values="views")

    fig = px.imshow(overview_matrix)
    fig.show()


if __name__ == "__main__":
    main()
