# localisation.py
import numpy as np
import plotly.graph_objects as go

# Example microphone positions for a hexagonal arrangement (in meters)
R = 1  # Distance from center to each microphone
MIC_POSITIONS = np.array([
    [R * np.cos(np.pi / 3 * i), R * np.sin(np.pi / 3 * i), 0]  # All microphones at Z = 0
    for i in range(6)
])

def locate_gunshot(toa):
    """Calculate the location of the gunshot based on Time of Arrival (ToA) data."""
    c = 343.0  # Speed of sound (m/s)
    distances = np.array(toa) * c

    # Use trilateration to find the position (x, y, z)
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(3):
        A[i] = 2 * (MIC_POSITIONS[i + 1] - MIC_POSITIONS[0])
        b[i] = distances[0]**2 - distances[i + 1]**2 + \
                MIC_POSITIONS[i + 1][0]**2 - MIC_POSITIONS[0][0]**2 + \
                MIC_POSITIONS[i + 1][1]**2 - MIC_POSITIONS[0][1]**2

    print("Matrix A:\n", A)
    print("Vector b:\n", b)

    # Check if the matrix is singular
    if np.linalg.det(A) == 0:
        print("Warning: Singular matrix detected! Returning None.")
        return None  # Handle the singular case

    location = np.linalg.solve(A, b)
    return location



def create_plotly_3d_plot(location, mic_positions):
    """Create a 3D plot of the gunshot location and microphone positions."""
    fig = go.Figure()

    # Add microphone positions
    fig.add_trace(go.Scatter3d(
        x=mic_positions[:, 0], y=mic_positions[:, 1], z=mic_positions[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Microphones'
    ))

    # Add gunshot location
    fig.add_trace(go.Scatter3d(
        x=[location[0]], y=[location[1]], z=[location[2]],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Gunshot Location'
    ))

    # Update layout
    fig.update_layout(scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='cube'
    ))

    plot_html = fig.to_html(full_html=False)
    return plot_html
