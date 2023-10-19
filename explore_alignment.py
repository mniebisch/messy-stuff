import numpy as np
import pandas as pd
import plotly.express as px
from numpy import typing as npt
from plotly.subplots import make_subplots
from scipy import spatial


def angle_between_vectors(a: npt.NDArray, b: npt.NDArray) -> float:
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    angle_rad = np.arccos(dot_product / (magnitude_a * magnitude_b))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def angle_direction(a: npt.NDArray, b: npt.NDArray) -> tuple[int, str]:
    cross_product = np.cross(a, b)
    if cross_product > 0:
        return 1, "counterclockwise"
    elif cross_product < 0:
        return -1, "clockwise"
    else:
        return 0, "parallel (no rotation needed)"


# Set up data
origin = [0, 0, 0]
point1 = [1, 1, 1]
point2 = [2, -1, 0]

vector1 = np.array(point1) - np.array(origin)
vector2 = np.array(point2) - np.array(origin)

ind_xy = [0, 1]
ind_xz = [0, 2]
ind_yz = [2, 1]

angle_xy = angle_between_vectors(vector1[ind_xy], vector2[ind_xy])
angle_yz = angle_between_vectors(vector1[ind_yz], vector2[ind_yz])
angle_xz = angle_between_vectors(vector1[ind_xz], vector2[ind_xz])

print("xy angle", angle_xy)
print("yz angle", angle_yz)
print("xz angle", angle_xz)

identity = np.diag(np.full(3, 1))
rotation = spatial.transform.Rotation.from_matrix(identity)

vector1 = vector1.reshape((1, 3))
vector2 = vector2.reshape((1, 3))

rotation, _ = rotation.align_vectors(vector1, vector2)

vector1_rotated = np.dot(vector1, rotation.as_matrix())

point1_rotated = vector1_rotated.flatten().tolist()


df_rotated = [
    vals + [label]
    for vals, label in zip(
        [origin, point1_rotated],
        ["v1_rot", "v1_rot"],
    )
]

df_raw = [
    vals + [label]
    for vals, label in zip([origin, point1, origin, point2], ["v1", "v1", "v2", "v2"])
]
df = pd.DataFrame(df_raw + df_rotated, columns=["x", "y", "z", "line"])

fig_xy = px.line(df, x="x", y="y", color="line")
fig_xz = px.line(df, x="x", y="z", color="line")
fig_yz = px.line(df, x="z", y="y", color="line")
fig_xyz = px.line_3d(df, x="x", y="y", z="z", color="line")

# Create subplots
fig = make_subplots(
    rows=5,
    cols=3,
    specs=[
        [
            {"type": "xy", "rowspan": 2},
            {"type": "xy", "rowspan": 2},
            {"type": "xy", "rowspan": 2},
        ],
        [None, None, None],
        [{"type": "scene", "colspan": 3, "rowspan": 3}, None, None],
        [None, None, None],
        [None, None, None],
    ],
    subplot_titles=[
        f"xy {np.round(angle_xy, 2)}",
        f"xz {np.round(angle_xz, 2)}",
        f"zy {np.round(angle_yz, 2)}",
        "xyz",
    ],
)

for trace_ind in range(len(fig_xy["data"])):
    fig.append_trace(fig_xy["data"][trace_ind], row=1, col=1)
for trace_ind in range(len(fig_xz["data"])):
    fig.append_trace(fig_xz["data"][trace_ind], row=1, col=2)
for trace_ind in range(len(fig_yz["data"])):
    fig.append_trace(fig_yz["data"][trace_ind], row=1, col=3)
for trace_ind in range(len(fig_xyz["data"])):
    fig.append_trace(fig_xyz["data"][trace_ind], row=3, col=1)

# Update layout
fig.update_layout(title="Explore alignment", height=1500, width=1500)

for col_ind, x_title, y_title in zip([1, 2, 3], ["x", "x", "z"], ["y", "z", "y"]):
    fig.update_xaxes(range=[-3, 3], row=1, col=col_ind, title_text=x_title)
    fig.update_yaxes(range=[-3, 3], row=1, col=col_ind, title_text=y_title)


# Show the figure
fig.show()
