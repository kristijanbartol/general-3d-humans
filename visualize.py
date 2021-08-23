import plotly.graph_objects as go
import numpy as np
import os


def draw(session_id, epoch_idx, iteration, dataset, pool_metrics):
    pose = pool_metrics.wavg.pose.detach().numpy()
    pose /= (pose.max() / 10.)
    pose = np.abs(pose)
    x, y, z = np.split(pose, pose.shape[1], axis=1)
    x, y, z = np.array([0.5]), np.array([0.5]), np.array([0.5])
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])

    dir_path = os.path.join('vis', f'{session_id}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # save figure
    fig.write_image(os.path.join(dir_path, f'{epoch_idx}_{iteration}_{dataset}.png'))


if __name__ == '__main__':
    # Helix equation
    t = np.linspace(0, 20, 100)
    x, y, z = np.cos(t), np.sin(t), t

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            color=z,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    #fig.show()
    fig.write_image('img.png')
