import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

# Set up data
origin = [0, 0, 0]
point1 = [1, 1, 1]
point2 = [2, -1, 0]

df_raw = [
    vals + [label]
    for vals, label in zip([origin, point1, origin, point2], ["v1", "v1", "v2", "v2"])
]
df = pd.DataFrame(df_raw, columns=["x", "y", "z", "line"])

fig_xy = px.line(df, x="x", y="y", color="line")
fig_xz = px.line(df, x="x", y="z", color="line")
fig_yz = px.line(df, x="z", y="y", color="line")
fig_xyz = px.line_3d(df, x="x", y="y", z="z", color="line")

# Create subplots
fig = make_subplots(
    rows=4,
    cols=3,
    specs=[
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [{"type": "scene", "colspan": 3, "rowspan": 3}, None, None],
        [None, None, None],
        [None, None, None],
    ],
    subplot_titles=["xy", "xz", "zy", "xyz"],
)

for trace_ind in range(len(fig_xy["data"])):
    fig.append_trace(fig_xy["data"][trace_ind], row=1, col=1)
for trace_ind in range(len(fig_xz["data"])):
    fig.append_trace(fig_xz["data"][trace_ind], row=1, col=2)
for trace_ind in range(len(fig_yz["data"])):
    fig.append_trace(fig_yz["data"][trace_ind], row=1, col=3)
for trace_ind in range(len(fig_xyz["data"])):
    fig.append_trace(fig_xyz["data"][trace_ind], row=2, col=1)

# Update layout
fig.update_layout(title="Explore alignment")

# Show the figure
fig.show()
