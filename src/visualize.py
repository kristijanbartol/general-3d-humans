# Author: Kristijan Bartol


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('/general-3d-humans/')
from src.const import CONNECTIVITY_DICT


JOINT_NAMES = {
    'human36m': 
        ['left foot', 'left knee', 'left hip', 'right hip', 'right knee', 'right foot', 
        'pelvis', 'spine', 'thorax', 'neck', 'left wrist', 'left elbow', 'left shoulder', 'right shoulder', 
        'right elbow', 'right wrist','head'
    ]
}


def store_hypotheses(epoch_idx, iteration, dataset, data_type, hpool):
    save_path_template = os.path.join('output/', dataset, data_type, '{:02d}_{:03d}_{}.npy')
    for hyp_idx, hyp_pose in enumerate(hpool.poses):
        np.save(save_path_template.format(epoch_idx, iteration, hyp_idx), hyp_pose.cpu().detach().numpy())
    
    np.save(save_path_template.format(epoch_idx, iteration, 'best'), hpool.best.pose.cpu().detach().numpy())
    np.save(save_path_template.format(epoch_idx, iteration, 'worst'), hpool.worst.pose.cpu().detach().numpy())
    np.save(save_path_template.format(epoch_idx, iteration, 'most'), hpool.most.pose.cpu().detach().numpy())
    np.save(save_path_template.format(epoch_idx, iteration, 'least') ,hpool.least.pose.cpu().detach().numpy())
    np.save(save_path_template.format(epoch_idx, iteration, 'random'), hpool.random.pose.cpu().detach().numpy())
    np.save(save_path_template.format(epoch_idx, iteration, 'avg'), hpool.avg.pose.cpu().detach().numpy())
    np.save(save_path_template.format(epoch_idx, iteration, 'wavg'), hpool.wavg.pose.cpu().detach().numpy())


def store_qualitative(session_id, epoch_idx, iteration, dataset, data_type, pool_metrics):
    dir_path = os.path.join('vis', f'{session_id}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fig_path = os.path.join(dir_path, f'qualit_{epoch_idx}_{iteration}_{dataset}_{data_type}')

    segments = CONNECTIVITY_DICT[dataset]
    joint_names = JOINT_NAMES['human36m']
    
    hyps_dict = pool_metrics.get_qualitative_hyps_dict()
    num_hyps = len(hyps_dict)

    #fig = make_subplots(cols=num_hyps, rows=1, 
    #    specs=[[{'type': 'scene'}] * num_hyps],
    #    subplot_titles=[str(x) for x in hyps_dict.keys()])

    fig = make_subplots(cols=6, rows=1, 
        specs=[[{'type': 'scene'}] * 6],
        subplot_titles=[str(x) for x in hyps_dict.keys()],
        shared_xaxes=True,
        shared_yaxes=True,
        # shared_zaxs=True
        )

    # subplot title size
    fig.update_annotations(font_size=60)



    for j, hyp_name in enumerate(hyps_dict):
        pose = hyps_dict[hyp_name]

        # plot joints
        if j == 0:
            for k in range(pose.shape[0]):
                fig.add_trace(go.Scatter3d(
                    x=[pose[k, 0]],
                    y=[pose[k, 1]],
                    z=[pose[k, 2]],
                    mode='markers',
                    name=joint_names[k],
                    marker=dict(
                        size=9,
                        color=px.colors.qualitative.Dark24[k]
                    )
                ), row=1, col=j + 1)
        else:
            for k in range(pose.shape[0]):
                fig.add_trace(go.Scatter3d(
                    x=[pose[k, 0]],
                    y=[pose[k, 1]],
                    z=[pose[k, 2]],
                    mode='markers',
                    name='NO',
                    marker=dict(
                        size=9,
                        color=px.colors.qualitative.Dark24[k]
                    )
                ), row=1, col=j + 1)

        for i in range(len(segments)):

            # plot lines
            fig.add_trace(go.Scatter3d(
                x=[pose[segments[i][0], 0], pose[segments[i][1], 0], None],
                y=[pose[segments[i][0], 1], pose[segments[i][1], 1], None],
                z=[pose[segments[i][0], 2], pose[segments[i][1], 2], None],
                mode='lines',
                name='segment',
                line=dict(
                    color='black',
                    width=5
                )
            ), row=1, col=j + 1)

    # remove traces from legend
    for trace in fig['data']: 
        if ('segment' in trace['name']) or ('NO' in trace['name']): 
            trace['showlegend'] = False



    # Tight layout.
    fig.update_layout(scene_aspectmode='data',
        height=1200, #, margin=dict(r=100, l=100, b=100, t=100))
        width=3500,
        legend=dict(
                font=dict(size=40),
                ),)

    # Save figure.
    fig.write_image(fig_path + '.png')
    fig.write_html(fig_path + '.html')


# TODO: Simplify these.

def store_overall_metrics(session_id, epoch_idx, dataset, data_type, global_metrics):
    dir_path = os.path.join('vis', f'{session_id}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fig_path = os.path.join(dir_path, f'overall_{epoch_idx}_{dataset}_{data_type}.png')

    metrics_dict = global_metrics.get_overall_metrics_dict()
    pd_metrics = pd.DataFrame(metrics_dict)

    plt.figure()
    sns.set_theme(style="darkgrid")
    fig = sns.barplot(data=pd_metrics)

    fig.get_figure().savefig(fig_path)

    # TODO: Fix problem with non-closed figures.
    plt.close(fig.get_figure())

    # TODO: Remove this from here.
    logs_dir = f'./logs/{session_id}'
    values = np.array([x[1] for x in metrics_dict.items()])
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    np.save(os.path.join(logs_dir, f'overall_{epoch_idx}_{dataset}_{data_type}.npy'), values)


def store_pose_prior_metrics(session_id, epoch_idx, dataset, data_type, global_metrics):
    dir_path = os.path.join('vis', f'{session_id}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fig_path = os.path.join(dir_path, f'pose_prior_{epoch_idx}_{dataset}_{data_type}.png')

    metrics = np.array(global_metrics.get_pose_prior_metrics_dict())

    #x = ['weight', 'avg', 'most', 'least', 'stoch', 'random', 'best', 'worst']
    x = np.arange(8)

    plt.bar(x + 0.00, metrics[0], width = 0.15)
    plt.bar(x + 0.15, metrics[1], width = 0.15)
    plt.bar(x + 0.30, metrics[2], width = 0.15)
    plt.bar(x + 0.45, metrics[3], width = 0.15)
    plt.bar(x + 0.60, metrics[4], width = 0.15)
    plt.bar(x + 0.75, metrics[5], width = 0.15)

    plt.yscale('log')

    plt.style.use('seaborn')

    plt.savefig(fig_path)
    plt.close()


def store_transfer_learning_metrics(session_id, epoch_idx, errors):
    dir_path = os.path.join('vis', f'{session_id}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fig_path = os.path.join(dir_path, f'transfer_{epoch_idx}.png')

    errors = np.array(errors)
    # TODO: Add names to plot
    x = np.arange(errors.shape[0])

    x = ['CMU1', 'CMU2', 'CMU3', 'CMU4', 'H36M']

    plt.figure()
    sns.set_theme(style="darkgrid")
    fig = sns.barplot(x, errors)

    fig.get_figure().savefig(fig_path)

    # TODO: Fix problem with non-closed figures.
    plt.close(fig.get_figure())
