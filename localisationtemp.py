# localisation.py
import numpy as np
import plotly.graph_objects as go

# Hexagonal arrangement of microphones (in meters)
R = 1  # Distance from center to each microphone
MIC_POSITIONS = np.array([
    [R * np.cos(np.pi / 3 * i), R * np.sin(np.pi / 3 * i), 0]  # All microphones at Z = 0
    for i in range(6)
])

def locate_gunshot(toa):
    """
    Calculate the location of the gunshot based on Time of Arrival (ToA) data.
    """
    c = 343.0  # Speed of sound in m/s
    distances = np.array(toa) * c

    # Use trilateration to estimate the position (x, y, z)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(3):
        A[i] = 2 * (MIC_POSITIONS[i + 1] - MIC_POSITIONS[0])
        b[i] = distances[0]**2 - distances[i + 1]**2 + \
                MIC_POSITIONS[i + 1][0]**2 - MIC_POSITIONS[0][0]**2 + \
                MIC_POSITIONS[i + 1][1]**2 - MIC_POSITIONS[0][1]**2

    location = np.linalg.solve(A, b)
    return location

def create_plotly_3d_plot(location, mic_positions):
    """
    Create a 3D plot of the gunshot location and microphone positions.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=mic_positions[:, 0], y=mic_positions[:, 1], z=mic_positions[:, 2],
        mode='markers', marker=dict(size=5, color='blue'), name='Microphones'
    ))
    fig.add_trace(go.Scatter3d(
        x=[location[0]], y=[location[1]], z=[location[2]],
        mode='markers', marker=dict(size=8, color='red'), name='Gunshot Location'
    ))
    fig.update_layout(scene=dict(
        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='cube'
    ))
    return fig.to_html(full_html=False)
