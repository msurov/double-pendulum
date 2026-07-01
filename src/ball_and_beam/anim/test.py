import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.transforms import Affine2D

fig, ax = plt.subplots()

# create six colored arcs
colors = ['r', 'g', 'b', 'c', 'm', 'y']
arcs = []

for i in range(6):
    arc = Arc(
        (0, 0),           # center
        width=2,
        height=2,
        theta1=60*i,
        theta2=60*(i+1),
        lw=5,
        color=colors[i]
    )
    ax.add_patch(arc)
    arcs.append(arc)

ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.show()