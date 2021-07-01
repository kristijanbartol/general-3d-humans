import torch.nn as nn


def create_score_nn(input_size=50*17, hidden_layer_sizes=[200, 300, 200]):
    layers = [
        nn.Linear(input_size, hidden_layer_sizes[0]),
        nn.ReLU6()
    ]
    for idx in range(len(hidden_layer_sizes) - 1):
        layers += [
            nn.Linear(hidden_layer_sizes[idx], hidden_layer_sizes[idx+1]),
            nn.ReLU6()
        ]
    layers += [
        nn.Linear(hidden_layer_sizes[-1], 1),
        nn.Sigmoid()
    ]

    return nn.Sequential(*layers)
