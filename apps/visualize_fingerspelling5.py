import itertools
import pathlib
from typing import Dict, List
import re

import numpy as np
from numpy import typing as npt
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import plotly.express as px
import plotly.graph_objs as go
from sklearn import metrics as sk_metrics
import yaml

from fmp.datasets.fingerspelling5 import utils


def load_dataset(dataset_root_dir: pathlib.Path, dataset_name: str) -> pd.DataFrame:
    csv_path = dataset_root_dir / dataset_name / f"{dataset_name}.csv"
    landmark_data = pd.read_csv(csv_path)
    landmark_data = landmark_data.loc[~landmark_data.isnull().any(axis=1)]
    landmark_data = landmark_data.reset_index(drop=True)
    return landmark_data


def load_metrics(metrics_root_dir: pathlib.Path, dataset_name: str) -> pd.DataFrame:
    metrics_scaled_file = metrics_root_dir / f"{dataset_name}_scaled.csv"
    metrics_unscaled_file = metrics_root_dir / f"{dataset_name}_unscaled.csv"
    metrics_scaled = pd.read_csv(metrics_scaled_file)
    metrics_unscaled = pd.read_csv(metrics_unscaled_file)

    scale_col = "scaled"
    metrics_scaled[scale_col] = True
    metrics_unscaled[scale_col] = False

    metrics = pd.concat([metrics_scaled, metrics_unscaled], axis=0).reset_index(
        drop=True
    )

    return metrics


def load_predictions(predictions_file_path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(predictions_file_path)


# TODO fix unused argument
def load_predictions_hparams(predictions_file_path: pathlib.Path) -> Dict:
    hparams_path = predictions_full_path.with_suffix(".yaml")
    with open(hparams_path, "r") as hparams_file:
        hparams = yaml.safe_load(hparams_file)
    return hparams


def load_training_datasplit(prediction_hparams: Dict) -> pd.DataFrame:
    training_dir = pathlib.Path(prediction_hparams["ckpt"]).parent.parent
    training_config_file = training_dir / "config.yaml"

    with open(training_config_file, "r") as file:
        training_config = yaml.safe_load(file)

    datasplit_file = training_config["data"]["datasplit_file"]

    return pd.read_csv(datasplit_file)


def compute_row_facet_spacing(num_facet_rows: int) -> float:
    default_row_spacing = 0.025

    if num_facet_rows == 0:
        return default_row_spacing
    elif 1 / num_facet_rows > default_row_spacing:
        return default_row_spacing
    else:
        return 1 / num_facet_rows


def create_metric_graph(metrics_data: pd.DataFrame) -> go.Figure:
    # TODO how to represent SCALE? radio button? scaled, unscaled, both -> if of as facet col
    # reserve color for splits?
    # TODO function that creates app layout
    # TODO function that adds main overview callback?
    # TODO function that adds overview letter buttons
    # TODO functions that adds overview variable buttons?
    # TODO where to put these functions? local module (in apps dir)? or in module (fmp)

    num_facet_rows = metrics_data["variable"].nunique()

    facet_row_spacing = compute_row_facet_spacing(num_facet_rows)
    fig = px.box(
        metrics_data,
        x="letter",
        y="value",
        facet_row="variable",
        color="split",
        facet_row_spacing=facet_row_spacing,
    )

    letters = sorted(metrics_data["letter"].unique())
    fig.update_xaxes(categoryorder="array", categoryarray=letters)
    return fig


def compute_confusion_matrix(predictions: pd.DataFrame) -> npt.NDArray:
    return sk_metrics.confusion_matrix(
        y_true=predictions["letter"],
        y_pred=predictions["predictions"],
        labels=utils.fingerspelling5.letters,
    )


def create_confusion_matrix_graph(confusion_matrix: npt.NDArray) -> go.Figure:
    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted", y="Groundtruth", color="Count"),
        x=utils.fingerspelling5.letters,
        y=utils.fingerspelling5.letters,
    )
    fig.update_layout(autosize=True)
    return fig


def calc_metrics_on_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split = df.loc[df["split"] == "train"]
    valid_split = df.loc[df["split"] == "valid"]

    metrics_agg_train, metrics_label_train = calc_metrics_on_split(
        y_true=train_split["letter"], y_pred=train_split["predictions"]
    )
    metrics_agg_valid, metrics_label_valid = calc_metrics_on_split(
        y_true=valid_split["letter"], y_pred=valid_split["predictions"]
    )

    metrics_agg_train["split"] = "train"
    metrics_label_train["split"] = "train"

    metrics_agg_valid["split"] = "valid"
    metrics_label_valid["split"] = "valid"

    metrics_agg = pd.concat([metrics_agg_train, metrics_agg_valid])
    metrics_label = pd.concat([metrics_label_train, metrics_label_valid])

    return metrics_agg, metrics_label


def calc_metrics_on_split(
    y_true: npt.NDArray, y_pred: npt.NDArray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.sort(np.unique(y_true))
    recall_label = sk_metrics.recall_score(y_true, y_pred, average=None, labels=labels)
    precision_label = sk_metrics.precision_score(
        y_true, y_pred, average=None, labels=labels
    )
    f1_label = sk_metrics.f1_score(y_true, y_pred, average=None, labels=labels)
    recall_micro = sk_metrics.recall_score(y_true, y_pred, average="micro")
    precision_micro = sk_metrics.precision_score(y_true, y_pred, average="micro")
    f1_micro = sk_metrics.f1_score(y_true, y_pred, average="micro")
    recall_macro = sk_metrics.recall_score(y_true, y_pred, average="macro")
    precision_macro = sk_metrics.precision_score(y_true, y_pred, average="macro")
    f1_macro = sk_metrics.f1_score(y_true, y_pred, average="macro")
    accuracy = sk_metrics.accuracy_score(y_true, y_pred)

    metrics_agg = pd.DataFrame(
        {
            "metric": [
                "recall_micro",
                "recall_macro",
                "precision_micro",
                "precision_macro",
                "f1_micro",
                "f1_macro",
                "accuracy",
            ],
            "value": [
                recall_micro,
                recall_macro,
                precision_micro,
                precision_macro,
                f1_micro,
                f1_macro,
                accuracy,
            ],
        }
    )
    metrics_label = pd.DataFrame(
        {
            "recall": recall_label,
            "precision": precision_label,
            "f1": f1_label,
            "label": labels,
        }
    )
    metrics_label = pd.melt(metrics_label, id_vars=["label"])
    return metrics_agg, metrics_label


def add_letter_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    letter: str,
    color: str,
    x_var: str,
    y_var: str,
) -> None:
    df_letter = df.loc[df["letter"] == letter]
    df_correct = df_letter.loc[df_letter["predictions"] == letter]
    df_wrong = df_letter.loc[df_letter["predictions"] != letter]
    fig.add_trace(
        go.Scatter(
            name=f"'{letter}' correct",
            x=df_correct[x_var],
            y=df_correct[y_var],
            mode="markers",
            marker=dict(color=color, opacity=0.5, size=8),
            text=df_correct["predictions"],
        )
    )
    fig.add_trace(
        go.Scatter(
            name=f"'{letter}' wrong",
            x=df_wrong[x_var],
            y=df_wrong[y_var],
            mode="text",
            text=df_wrong["predictions"],
            textposition="middle center",
            textfont=dict(family="sans serif", size=10, color=color),
        )
    )


# Load data
root_path = pathlib.Path(__file__).parent.parent

metrics_path = root_path / "metrics"
data_path = root_path / "data" / "fingerspelling5"
predictions_path = root_path / "predictions"

dataset_name = "fingerspelling5_dummy"

# TODO save hparams for prediction similar to metric computation
# TODO add dataset name to pred filename? or read from yaml?
# is going to change the most?
predictions_filename = "prediction__version_0__epoch=8-step=9.csv"
predictions_full_path = predictions_path / "example" / predictions_filename
# predictions_file = predictions_path / "prediction__version_22__epoch=17-step=36.csv"

fingerspelling_data = load_dataset(data_path, dataset_name)
metrics = load_metrics(metrics_path, dataset_name)
predictions = load_predictions(predictions_full_path)

predictions_hparams = load_predictions_hparams(predictions_full_path)
training_datasplit = load_training_datasplit(predictions_hparams)
training_datasplit = training_datasplit.reset_index(names=["batch_indices"])

# Prepare data for plotting
# TODO use LIT data module instead!!!!!!!!1
letter_batch_index_map = pd.DataFrame(fingerspelling_data["letter"])
letter_batch_index_map["batch_indices"] = np.arange(len(letter_batch_index_map))

metrics = pd.merge(metrics, letter_batch_index_map, on="batch_indices", how="left")

metrics = pd.merge(
    metrics,
    training_datasplit[["batch_indices", "split"]],
    how="left",
    on="batch_indices",
)

metrics_long = pd.melt(metrics, id_vars=["batch_indices", "letter", "scaled", "split"])

# Process predictions
predictions["predictions"] = predictions["predictions"].replace(
    {ind: letter for ind, letter in enumerate(utils.fingerspelling5.letters)}
)
predictions = pd.merge(
    predictions,
    training_datasplit[["batch_indices", "split", "letter"]],
    on="batch_indices",
    how="left",
)

metrics = pd.merge(
    metrics,
    predictions[["batch_indices", "predictions"]],
    how="left",
    on="batch_indices",
)


# Compute confusion matrices
confusion_matrix_train = compute_confusion_matrix(
    predictions.loc[predictions["split"] == "train"]
)
confusion_matrix_valid = compute_confusion_matrix(
    predictions.loc[predictions["split"] == "valid"]
)


# TODO variable filtering


# Plotting

metric_options = sorted(metrics_long["variable"].unique())


def match_metric_regex(pattern: str) -> List[str]:
    return [option for option in metric_options if re.match(pattern, option)]


fig_cf_matrix_train = create_confusion_matrix_graph(confusion_matrix_train)
fig_cf_matrix_valid = create_confusion_matrix_graph(confusion_matrix_valid)

dist_orders = {
    "letter": utils.fingerspelling5.letters,
    "person": sorted(training_datasplit["person"].unique().tolist()),
}

dist_plots = {
    "person_label": px.histogram(
        training_datasplit,
        x="letter",
        color="person",
        pattern_shape="split",
        category_orders=dist_orders,
        barmode="group",
    ),
    "person": px.histogram(
        training_datasplit,
        x="person",
        color="split",
        category_orders=dist_orders,
    ),
    "label": px.histogram(
        training_datasplit,
        x="letter",
        color="split",
        category_orders=dist_orders,
        barmode="group",
    ),
}

# Create dropdown selection
letters = utils.fingerspelling5.letters
dist_plot_options = list(dist_plots.keys())

# Prediction Eval
metrics_agg, metrics_label = calc_metrics_on_splits(predictions)


fig_metrics_agg = px.bar(
    metrics_agg, x="metric", y="value", color="split", barmode="group"
)
fig_metrics_label = px.bar(
    metrics_label,
    x="label",
    y="value",
    color="split",
    facet_row="variable",
    barmode="group",
)

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
                        html.Button("empty", id="clear_letters_button", n_clicks=0),
                        html.Button("(m, n)", id="mn_button", n_clicks=0),
                        html.Button("(r, u, v)", id="ruv_button", n_clicks=0),
                        html.Button("all", id="all_button", n_clicks=0),
                        dcc.Dropdown(
                            metric_options,
                            [metric_options[0]],
                            id="overview_variable_picks",
                            multi=True,
                        ),
                        html.Button("empty", id="clear_variables_button", n_clicks=0),
                        html.Button("loc", id="loc_only_button", n_clicks=0),
                        html.Button("loc.*mean", id="loc_mean_button", n_clicks=0),
                        html.Button("loc.*std", id="loc_std_button", n_clicks=0),
                        html.Button("loc.*extend", id="loc_extend_button", n_clicks=0),
                        html.Button("space", id="space_only_button", n_clicks=0),
                        html.Button("space.*area", id="space_area_button", n_clicks=0),
                        html.Button(
                            "space.*perimeter", id="space_perimeter_button", n_clicks=0
                        ),
                        html.Button("dist", id="dist_only_button", n_clicks=0),
                        html.Button("dist.*mean", id="dist_mean_button", n_clicks=0),
                        html.Button("dist.*std", id="dist_std_button", n_clicks=0),
                        html.Button("angle", id="angle_only_button", n_clicks=0),
                        dcc.Dropdown(
                            ["scaled", "raw"], "scaled", id="scale_flag_overview"
                        ),
                        dcc.Graph(
                            id="graph_overview",
                            style={"width": "99vw", "height": "80vh"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="confusion matrix",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(
                                        id="confusion_matrix_train",
                                        figure=fig_cf_matrix_train,
                                        style={"height": "100%", "width": "100%"},
                                    ),
                                    style={
                                        "display": "inline-block",
                                        "width": "48%",
                                        "height": "80vh",
                                    },
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="confusion_matrix_valid",
                                        figure=fig_cf_matrix_valid,
                                        style={"height": "100%", "width": "100%"},
                                    ),
                                    style={
                                        "display": "inline-block",
                                        "width": "48%",
                                        "height": "80vh",
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="scatter",
                    children=[
                        dcc.Dropdown(metric_options, metric_options[0], id="x_dim"),
                        dcc.Dropdown(metric_options, metric_options[1], id="y_dim"),
                        dcc.Dropdown(
                            letters, letters[0], id="letter_picks", multi=True
                        ),
                        dcc.Dropdown(
                            ["scaled", "raw"], "scaled", id="scale_flag_scatter"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(
                                        id="scatter_graph_train",
                                        style={"height": "100%", "width": "100%"},
                                    ),
                                    style={
                                        "display": "inline-block",
                                        "width": "48%",
                                        "height": "80vh",
                                    },
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="scatter_graph_valid",
                                        style={"height": "100%", "width": "100%"},
                                    ),
                                    style={
                                        "display": "inline-block",
                                        "width": "48%",
                                        "height": "80vh",
                                    },
                                ),
                            ]
                        ),
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
                        dcc.Graph(id="pred_metrics_agg_graph", figure=fig_metrics_agg),
                        dcc.Graph(
                            id="pred_metrics_label_graph", figure=fig_metrics_label
                        ),
                    ],
                ),
            ]
        )
    ]
)


@app.callback(
    Output(component_id="scatter_graph_train", component_property="figure"),
    Input(component_id="x_dim", component_property="value"),
    Input(component_id="y_dim", component_property="value"),
    Input(component_id="letter_picks", component_property="value"),
    Input(component_id="scale_flag_scatter", component_property="value"),
)
def update_scatter_train(
    x_dim: str, y_dim: str, letter_picks: list[str], scale_flag_scatter: str
):
    fig = go.Figure()
    for letter_pick, color in zip(
        letter_picks, itertools.cycle(px.colors.qualitative.T10)
    ):
        metrics_wide_filtered = metrics.loc[metrics["letter"] == letter_pick]
        metrics_wide_filtered = metrics_wide_filtered.loc[
            metrics_wide_filtered["split"] == "train"
        ]
        metrics_wide_filtered = metrics_wide_filtered.loc[
            metrics_wide_filtered["scaled"] == (scale_flag_scatter == "scaled")
        ]
        add_letter_trace(fig, metrics_wide_filtered, letter_pick, color, x_dim, y_dim)

    x_values = metrics.loc[metrics["scaled"] == (scale_flag_scatter == "scaled"), x_dim]
    y_values = metrics.loc[metrics["scaled"] == (scale_flag_scatter == "scaled"), y_dim]

    y_min, y_max = y_values.min(), y_values.max()
    x_min, x_max = x_values.min(), x_values.max()

    fig.update_yaxes(range=[y_min, y_max])
    fig.update_xaxes(range=[x_min, x_max])

    return fig


@app.callback(
    Output(component_id="scatter_graph_valid", component_property="figure"),
    Input(component_id="x_dim", component_property="value"),
    Input(component_id="y_dim", component_property="value"),
    Input(component_id="letter_picks", component_property="value"),
    Input(component_id="scale_flag_scatter", component_property="value"),
)
def update_scatter_valid(
    x_dim: str, y_dim: str, letter_picks: list[str], scale_flag_scatter: str
):
    fig = go.Figure()
    for letter_pick, color in zip(
        letter_picks, itertools.cycle(px.colors.qualitative.T10)
    ):
        metrics_wide_filtered = metrics.loc[metrics["letter"] == letter_pick]
        metrics_wide_filtered = metrics_wide_filtered.loc[
            metrics_wide_filtered["split"] == "valid"
        ]
        metrics_wide_filtered = metrics_wide_filtered.loc[
            metrics_wide_filtered["scaled"] == (scale_flag_scatter == "scaled")
        ]
        add_letter_trace(fig, metrics_wide_filtered, letter_pick, color, x_dim, y_dim)

    x_values = metrics.loc[metrics["scaled"] == (scale_flag_scatter == "scaled"), x_dim]
    y_values = metrics.loc[metrics["scaled"] == (scale_flag_scatter == "scaled"), y_dim]

    y_min, y_max = y_values.min(), y_values.max()
    x_min, x_max = x_values.min(), x_values.max()

    fig.update_yaxes(range=[y_min, y_max])
    fig.update_xaxes(range=[x_min, x_max])

    return fig


@app.callback(
    Output(component_id="dist_graph", component_property="figure"),
    Input(component_id="dist_option", component_property="value"),
)
def update_dist(dist_option: str):
    return dist_plots[dist_option]


@app.callback(
    Output(component_id="graph_overview", component_property="figure"),
    Input(component_id="overview_letter_picks", component_property="value"),
    Input(component_id="overview_variable_picks", component_property="value"),
    Input(component_id="scale_flag_overview", component_property="value"),
)
def update_overview(
    overview_letter_picks: list[str],
    overview_variable_picks: list[str],
    scale_flag_overview: str,
):
    # pick scaled or raw data
    metrics_filtered = metrics_long.loc[
        metrics_long["scaled"] == (scale_flag_overview == "scaled")
    ]
    # filter variables
    variable_mask = metrics_filtered["variable"].isin(overview_variable_picks)
    metrics_filtered = metrics_filtered.loc[variable_mask]

    # filter letters
    metrics_filtered = metrics_filtered.loc[
        metrics_filtered["letter"].isin(overview_letter_picks)
    ]

    fig_overview = create_metric_graph(metrics_filtered)

    return fig_overview


@app.callback(
    Output(
        "overview_letter_picks",
        "value",
        allow_duplicate=True,
    ),
    Input("mn_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_mn_letters(value):
    return ["m", "n"]


@app.callback(
    Output("overview_letter_picks", "value", allow_duplicate=True),
    Input("ruv_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_ruv_letters(value):
    return ["r", "u", "v"]


@app.callback(
    Output("overview_letter_picks", "value", allow_duplicate=True),
    Input("clear_letters_button", "n_clicks"),
    prevent_initial_call=True,
)
def clear_letters(value):
    return []


@app.callback(
    Output("overview_letter_picks", "value", allow_duplicate=True),
    Input("all_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_all_letters(value):
    return letters


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("loc_only_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_loc_variable(value):
    return match_metric_regex("loc")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("loc_mean_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_loc__mean_variable(value):
    return match_metric_regex("loc.*mean")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("loc_std_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_loc_std_variable(value):
    return match_metric_regex("loc.*std")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("loc_extend_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_loc_extend_variable(value):
    return match_metric_regex("loc.*extend")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("space_only_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_sppace_variable(value):
    return match_metric_regex("space")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("space_area_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_space_area_variable(value):
    return match_metric_regex("space.*area")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("space_perimeter_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_space_perimeter_variable(value):
    return match_metric_regex("space.*perimeter")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("dist_only_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_dist_variable(value):
    return match_metric_regex("dist")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("dist_mean_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_dist_mean_variable(value):
    return match_metric_regex("dist.*mean")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("dist_std_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_dist_std_variable(value):
    return match_metric_regex("dist.*std")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("angle_only_button", "n_clicks"),
    prevent_initial_call=True,
)
def set_angle_variable(value):
    return match_metric_regex("angle")


@app.callback(
    Output("overview_variable_picks", "value", allow_duplicate=True),
    Input("clear_variables_button", "n_clicks"),
    prevent_initial_call=True,
)
def clear_variables(value):
    return []


if __name__ == "__main__":
    app.run(debug=True)
    print("Done")
