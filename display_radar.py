# display_radar.py
import matplotlib.pyplot as plt
import numpy as np

def display_radar(location):
    """
    Display radar-like visualization for the gunshot's direction.
    """
    x, y = location[0], location[1]
    angle = np.arctan2(y, x)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot([0, angle], [0, 1], marker='o', color='r')
    ax.set_rticks([])  # Hide radial ticks
    ax.set_rmax(1.5)   # Max radius for radar
    plt.show()

# Example call (replace with your gunshot location data)
# location = locate_gunshot([0.0012, 0.0014, 0.0013, 0.0011, 0.0015, 0.0016])
# display_radar(location)
