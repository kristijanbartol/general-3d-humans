import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os

from metrics import center_pelvis


def __draw_pose(pose, fig_path):
    # Prepare data.
    x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]

    # Create figure.
    fig = go.Figure(data=[
        go.Scatter3d(x=x, y=y, z=z, mode='markers+text', text=[str(x) for x in range(pose.shape[0])])])
    
    # Tight layout.
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Save figure.
    fig.write_image(fig_path)


def create_fig_path(dir_path, epoch_idx, iteration, dataset, suffix):
    return os.path.join(dir_path, f'{epoch_idx}_{iteration}_{dataset}_{suffix}.png')


def draw(session_id, epoch_idx, iteration, dataset, pool_metrics):
    dir_path = os.path.join('vis', f'{session_id}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    est_pose = center_pelvis(pool_metrics.wavg.pose.detach().numpy())
    gt_pose = center_pelvis(pool_metrics.gt_3d.detach().numpy())
    est_pose = est_pose

    __draw_pose(est_pose, create_fig_path(dir_path, epoch_idx, iteration, dataset, 'est'))
    __draw_pose(gt_pose, create_fig_path(dir_path, epoch_idx, iteration, dataset, 'gt'))

    '''
    # Prepare data.
    est_x, est_y, est_z = est_pose[:, 0], est_pose[:, 1], est_pose[:, 2]
    gt_x, gt_y, gt_z = gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2]

    # Create figure.
    fig = go.Figure(data=[
        go.Scatter3d(x=est_x, y=est_y, z=est_z, mode='markers+text', text=[str(x) for x in range(17)])])
    #fig = make_subplots(cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]])
    
    #fig.add_trace(go.Scatter3d(x=est_x, y=est_y, z=est_z, mode='markers'),
    #          row=1, col=1)
    #fig.add_trace(go.Scatter3d(x=gt_x, y=gt_y, z=gt_z, mode='markers'),
    #    row=1, col=2)
    
    # Tight layout.
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Save figure.
    fig.write_image(os.path.join(dir_path, f'{epoch_idx}_{iteration}_{dataset}_est.png'))

    fig = go.Figure(data=[go.Scatter3d(x=gt_x, y=gt_y, z=gt_z, mode='markers+text', text=[str(x) for x in range(17)])])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.write_image(os.path.join(dir_path, f'{epoch_idx}_{iteration}_{dataset}_gt.png'))
    '''


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