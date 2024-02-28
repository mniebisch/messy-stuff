import itertools
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tqdm
from dash import Dash, Input, Output, dcc, html
from numpy import typing as npt
from sklearn import metrics

import hand_description
import pipeline_fingerspelling5
from cam_mediapipe_singlehand import number_to_letter


def reshape_hands(df: pd.DataFrame) -> npt.NDArray:
    cols = pipeline_fingerspelling5.generate_hand_landmark_columns()
    hand_vals = df[cols].values
    hand_vals = hand_vals.reshape((-1, 21, 3))
    return hand_vals.astype(np.float64)


# load dataeset -> labels and co contained?
data_path = pathlib.Path(__file__).parent / "data"
fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"
landmark_data = pd.read_csv(fingerspelling_landmark_csv)

# train/test/valid split?

# are there nans?
landmark_data = landmark_data.loc[~landmark_data.isnull().any(axis=1)]
landmark_data = landmark_data.reset_index()

# scaling of landmarks?

# transform
hands = reshape_hands(landmark_data)

# create fake preds
n_samples = hands.shape[0]
pred_pool = [number_to_letter(i) for i in range(24)]
preds = np.random.choice(pred_pool, size=n_samples)
preds = pd.Series(preds, name="preds")

# compute stats
extent_stats = [hand_description.compute_extend(hand) for hand in tqdm.tqdm(hands)]
extent_stats = pd.DataFrame(extent_stats, columns=["x_extent", "y_extent", "z_extent"])

stats = pd.concat([landmark_data[["letter"]], extent_stats, preds], axis=1)

letters = sorted(stats["letter"].unique().tolist())

id_vars = ("letter", "preds")  # Add split here later
stats_long = pd.melt(stats, id_vars=id_vars)

var_columns = set(stats.columns) - set(id_vars)
var_columns = list(var_columns)

# compute confusion matrix
confusion_matrix = metrics.confusion_matrix(
    y_true=stats["letter"], y_pred=stats["preds"], labels=letters
)
fig_cf = px.imshow(
    confusion_matrix,
    labels=dict(x="Predicted", y="Groundtruth", color="Count"),
    x=letters,
    y=letters,
)


def add_letter_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    letter: str,
    color: str,
    x_var: str,
    y_var: str,
) -> None:
    df_letter = df.loc[df["letter"] == letter]
    df_correct = df_letter.loc[df_letter["preds"] == letter]
    df_wrong = df_letter.loc[df_letter["preds"] != letter]
    fig.add_trace(
        go.Scatter(
            name=f"'{letter}' correct",
            x=df_correct[x_var],
            y=df_correct[y_var],
            mode="markers",
            marker=dict(color=color, opacity=0.5, size=8),
            text=df_correct["preds"],
        )
    )
    fig.add_trace(
        go.Scatter(
            name=f"'{letter}' wrong",
            x=df_wrong[x_var],
            y=df_wrong[y_var],
            mode="text",
            text=df_wrong["preds"],
            textposition="middle center",
            textfont=dict(family="sans serif", size=10, color=color),
        )
    )


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
                        dcc.Graph(id="confusion_matrix", figure=fig_cf),
                    ],
                ),
                dcc.Tab(
                    label="scatter",
                    children=[
                        dcc.Dropdown(var_columns, var_columns[0], id="x_dim"),
                        dcc.Dropdown(var_columns, var_columns[1], id="y_dim"),
                        dcc.Dropdown(
                            letters, letters[0], id="letter_picks", multi=True
                        ),
                        dcc.Graph(id="graph"),
                    ],
                ),
            ]
        )
    ]
)


@app.callback(
    Output(component_id="graph", component_property="figure"),
    Input(component_id="x_dim", component_property="value"),
    Input(component_id="y_dim", component_property="value"),
    Input(component_id="letter_picks", component_property="value"),
)
def update_scatter(x_dim: str, y_dim: str, letter_picks: list[str]):
    fig = go.Figure()
    for letter_pick, color in zip(
        letter_picks, itertools.cycle(px.colors.qualitative.T10)
    ):
        stats_wide_filtered = stats.loc[stats["letter"] == letter_pick]
        add_letter_trace(fig, stats_wide_filtered, letter_pick, color, x_dim, y_dim)

    return fig


@app.callback(
    Output(component_id="graph_overview", component_property="figure"),
    Input(component_id="overview_letter_picks", component_property="value"),
)
def update_overview(overview_letter_picks: list[str]):
    stats_long_filtered = stats_long.loc[
        stats_long["letter"].isin(overview_letter_picks)
    ]
    # TODO setting color to 'letter' is a temporary fix for column alignment of the boxplot
    #  in the future color will be user for 'split'
    fig_overview = px.box(
        stats_long_filtered, x="letter", y="value", facet_row="variable", color="letter"
    )
    fig_overview.update_xaxes(
        categoryorder="array", categoryarray=sorted(overview_letter_picks)
    )

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
    Input("clear_button", "n_clicks"),
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


if __name__ == "__main__":
    app.run(debug=True)
    print("Done")

# TODO add tabs https://dash.plotly.com/dash-core-components/tabs
# TODO add confusion matrices (tab of its own)
