import torch.nn as nn


def create_camera_nn(input_size=50*17, hidden_layer_sizes=[700, 500, 300, 300, 100], out_size=1):
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
        nn.Linear(hidden_layer_sizes[-1], out_size),
        nn.Sigmoid()
        #nn.ReLU()
        #nn.Softmax()
    ]

    return nn.Sequential(*layers)


# TODO: Add residual blocks (check simple-yet-effective paper).
# TODO: Check SYE paper to see how they normalize inputs.
def create_pose_nn(input_size=17*3, hidden_layer_sizes=[50, 50, 50], out_size=1):
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
        nn.Linear(hidden_layer_sizes[-1], out_size),
        nn.Sigmoid()
        #nn.ReLU()
        #nn.Softmax()
    ]

    return nn.Sequential(*layers)
