import plotly.graph_objects as go
import numpy as np

# 1. Create the base data for the x-axis
# This will remain constant throughout the animation
x = np.linspace(0, 2 * np.pi, 100)

# 2. Create the initial figure
# We start with the first frame of our animation (amplitude = 1)
initial_y = np.sin(x)

fig = go.Figure(
    data = [go.Scatter(x=x, y=initial_y, mode='lines')],
    layout = go.Layout(
        title="Sine Wave with Increasing Amplitude",
        xaxis_title = "Angle (radians)",
        yaxis_title = "Value",
        # Set a fixed y-axis range so the plot doesn't "jump"
        yaxis=dict(range=[-5, 5])
    )
)

fig.write_html('dump/index.html')
fig.write_image('dump/img.png')

print("Animated HTML file 'sine_wave_animation.html' has been created.")