# Generalizable Human Pose Triangulation

## Introduction

This is the source code for the paper "Generalizable Human Pose Triangulation".
It contains the instructions on installation and demos for training and testing
the presented model, as well as for estimating relative camera poses. We include
the neccessary data needed to run the demos. However, it is probably required to
first build a docker image and run the container --- docker instructions are also
provided.

## Usage

To install and prepare the environment, use docker:

```
docker build -t <image-name> .

docker run --rm --gpus all --name <container-name> -it \
	-v ${REPO_DIR}:/generalizable-triangulation \
	-v ${BASE_DATA_DIR}/:/data/  <image-name>
```


### Training

To train on the base configuration (use Human3.6M for training), run:

```
python src/main.py \\
	--posedsac_only \\
	--transfer -1 \\
	--temp 1.8 \\
	--gumbel \\
	--entropy_beta_cam .01 \\
	-lr 0.0005 \\
	-lrs 10 \\
	-ts 5 \\
	--temp_gamma 0.9 \\
	--train_iterations 10 \\
	--valid_iterations 1 \\
	--test_iterations 40 
	--pose_hypotheses 200 \\
	--layers_posedsac 1000 900 900 900 900 700 \\
	--entropy_beta_pose 0.01 \\
	--est_beta 0.02 \\
	--exp_beta 1. \\
	--body_lengths_mode 2 \\ 
	--pose_batch_size 16
```

To train on the intra-dataset configuration (specify the CMU set, from 0-3, using transfer argument), run:

```
python src/main.py \\
	--posedsac_only \\
	--transfer <cmu_set> \\
	--temp 1.8 \\
	--gumbel \\
	--entropy_beta_cam .01 \\
	-lr 0.0005 \\
	-lrs 10 \\
	-ts 5 \\
	--temp_gamma 0.9 \\
	--train_iterations 10 \\
	--valid_iterations 1 \\
	--test_iterations 40 
	--pose_hypotheses 200 \\
	--layers_posedsac 1000 900 900 900 900 700 \\
	--entropy_beta_pose 0.01 \\
	--est_beta 0.02 \\
	--exp_beta 1. \\
	--body_lengths_mode 2 \\ 
	--pose_batch_size 16
```

To train on the inter-dataset configuration (specify the CMU set, from 0-3, using transfer argument), run:

```
python src/main.py \\
	--posedsac_only \\
	--transfer <cmu_set> \\
	--temp 1.8 \\
	--gumbel \\
	--entropy_beta_cam .01 \\
	-lr 0.0005 \\
	-lrs 10 \\
	-ts 5 \\
	--temp_gamma 0.9 \\
	--train_iterations 10 \\
	--valid_iterations 1 \\
	--test_iterations 40 
	--pose_hypotheses 200 \\
	--layers_posedsac 1000 900 900 900 900 700 \\
	--entropy_beta_pose 0.01 \\
	--est_beta 0.02 \\
	--exp_beta 1. \\
	--body_lengths_mode 2 \\ 
	--pose_batch_size 16
```

A more convenient way to specify the arguments is through the .vscode/launch.json, if the VSCode IDE is used.


### Testing




### Results

| Base        |    Intra    | Inter |
| --- | --- |
| 29.1      | 25.6       | 31.0 |


### Data and pretrained models

The data used for the above commands is in ./data/ folder, the pretrained
models are provided in ./models/ folder.
