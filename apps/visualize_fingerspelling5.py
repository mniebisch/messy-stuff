import pathlib
from typing import List

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from fmp.datasets.fingerspelling5 import utils


def extract_metric_columns(
    non_variable_cols: List[str], metric_df_cols: List[str]
) -> List[str]:
    metric_cols = set(metric_df_cols) - set(non_variable_cols)
    return list(metric_cols)


# Load data
root_path = pathlib.Path(__file__).parent.parent

metrics_path = root_path / "metrics"
data_path = root_path / "data"
predictions_path = root_path / "predictions"

metrics_scaled_file = metrics_path / "fingerspelling5_singlehands_scaled.csv"
metrics_unscaled_file = metrics_path / "fingerspelling5_singlehands.csv"
fingerspelling_file = data_path / "fingerspelling5_singlehands.csv"
# is going to change the most?
predictions_file = predictions_path / "prediction__version_22__epoch=17-step=36.csv"


metrics_scaled = pd.read_csv(metrics_scaled_file)
metrics_unscaled = pd.read_csv(metrics_unscaled_file)

scale_col = "scaled"
metrics_scaled[scale_col] = True
metrics_unscaled[scale_col] = False

metrics = pd.concat([metrics_scaled, metrics_unscaled], axis=0).reset_index(drop=True)

fingerspelling_data = pd.read_csv(fingerspelling_file)
fingerspelling_data = fingerspelling_data.loc[~fingerspelling_data.isnull().any(axis=1)]
fingerspelling_data = fingerspelling_data.reset_index(drop=True)


predictions = pd.read_csv(predictions_file)

# Plotting

dist_plots = {
    "person_label": None,
    "person": None,
    "label": None,
}

# Create dropdown selection
letters = utils.fingerspelling5.letters
metric_cols = extract_metric_columns(
    non_variable_cols=["batch_indices", "scaled"],
    metric_df_cols=metrics.columns.tolist(),
)
dist_plot_options = list(dist_plots.keys())
# App layout
app = Dash(__name__)
app.layout = html.Div(
    [
        dcc.Tabs(
            [
                dcc.Tab(
                    label="overview",
                    children=[
                        dcc.Dropdown(
                            letters, letters, id="overview_letter_picks", multi=True
                        ),
                        html.Button("empty", id="clear_button", n_clicks=0),
                        html.Button("(m, n)", id="mn_button", n_clicks=0),
                        html.Button("(r, u, v)", id="ruv_button", n_clicks=0),
                        html.Button("all", id="all_button", n_clicks=0),
                        dcc.Graph(id="graph_overview"),
                    ],
                ),
                dcc.Tab(
                    label="confusion matrix",
                    children=[
                        dcc.Graph(id="confusion_matrix", figure=None),
                    ],
                ),
                dcc.Tab(
                    label="scatter",
                    children=[
                        dcc.Dropdown(metric_cols, metric_cols[0], id="x_dim"),
                        dcc.Dropdown(metric_cols, metric_cols[1], id="y_dim"),
                        dcc.Dropdown(
                            letters, letters[0], id="letter_picks", multi=True
                        ),
                        dcc.Graph(id="graph"),
                    ],
                ),
                dcc.Tab(
                    label="label_dist",
                    children=[
                        dcc.Dropdown(
                            list(dist_plots.keys()), "person_label", id="dist_option"
                        ),
                        dcc.Graph(id="dist_graph"),
                    ],
                ),
                dcc.Tab(
                    label="pred metrics",
                    children=[
                        dcc.Graph(id="pred_metrics_agg_graph", figure=None),
                        dcc.Graph(id="pred_metrics_label_graph", figure=None),
                    ],
                ),
            ]
        )
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
    print("Done")
