import plotly.graph_objects as go
import numpy as np

x = np.linspace(0, 1, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig = go.Figure()
fig.add_scatter(x=x, y=y1, mode="lines", name="y1", line=dict(width=2, color="black"))
fig.add_scatter(x=x, y=y2, mode="lines", name="y2")
fig.update_layout(
    margin=dict(l=40, r=10, t=10, b=40),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=12),
    showlegend=False,
)
fig.update_xaxes(
    showline=True,
    linewidth=1,
    linecolor="black",
    mirror=True,
    ticks="outside",
    showgrid=True,
    gridcolor="rgba(0,0,0,0.2)",
)
fig.update_yaxes(
    showline=True,
    linewidth=1,
    linecolor="black",
    mirror=True,
    ticks="outside",
    showgrid=True,
    gridcolor="rgba(0,0,0,0.2)",
)
fig.write_image('dump/img.png')
fig.write_html('dump/index.html', include_plotlyjs='cdn')