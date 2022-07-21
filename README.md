# Generalizable Human Pose Triangulation

## Introduction

This is the source code for the CVPR 2022 paper [Generalizable Human Pose Triangulation](https://arxiv.org/abs/2110.00280).
The repository will soon contain the complete instructions and demos for training and inference of
the models. NOTE: There is still a bit more work to do to make scripts more user-friendly. I apologize for that - feel free to reach
out if you encounter any issues!

<img src="https://github.com/kristijanbartol/general-humans/blob/main/assets/transfer-learning-fig.png" width="650">

## Citation

If you use our model in your research, please reference our paper:

```
@inproceedings{Bartol:CVPR:2022,
   title = {Generalizable Human Pose Triangulation},
   author = {Bartol, Kristijan and Bojani\'{c}, David and Petkovi\'{c}, Tomislav and Pribani\'{c}, Tomislav},
   booktitle = {Proceedings of IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
   month = jun,
   year = {2022}
}
```

## Updates / Work-In-Progress

We plan to completely prepare the source code with the pretrained models, demos, and videos by mid May. The to-do list consists of:

- [X] [19-04-2022] Instructions for training pose estimation model
- [X] [19-04-2022] Fundamental matrix estimation algorithm
- [X] [22-04-2022] Refactor the source code
- [ ] Complete the documentation
- [X] [26-04-2022] Pretrained pose estimation learning model
- [ ] Pretrained fundamental matrix estimation learning model
- [X] [28-04-2022] Instructions for running inference
- [X] Demo (main) scripts to run training and evaluation
- [ ] Video demonstration

## Usage

First download pretrained [backbone](https://github.com/karfly/learnable-triangulation-pytorch#model-zoo) and place it in ./models/pretrained/.

To install and prepare the environment, use docker:

```
docker build -t <image-name> .

docker run --rm --gpus all --name <container-name> -it \
	-v ${REPO_DIR}:/generalizable-triangulation \
	-v ${BASE_DATA_DIR}/:/data/  <image-name>
```

### Data preparation

Prior to running any training/evaluation/inference, 2D pose detections need to be extracted. Our backmode 2D pose detector is [the baseline model](https://github.com/microsoft/human-pose-estimation.pytorch), i.e., the version available in [karfly/learnable-triangulation-pytorch](https://github.com/karfly/learnable-triangulation-pytorch#model-zoo), but the straightforward inference method is not provided so it's not straightforward to use it. Instead, but with no guarantees, pose detectors such as [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) or [MMPose](https://github.com/open-mmlab/mmpose) can be used.

But we already prepared some training/evaluation [data](https://ferhr-my.sharepoint.com/:f:/g/personal/kbartol_fer_hr/Elo0eBbuQqhDmMYCypM5W3gBCYdxHiXIeavJ6lvzGzGMfg?e=BmXjV7) :) (password: _data-3d-humans_). Extract the folder into `data/<dataset>`. Note that the Human3.6M dataset already contains bounding boxes obtained as described [here](https://github.com/karfly/learnable-triangulation-pytorch/tree/master/mvn/datasets/human36m_preprocessing).

### Pose estimation model training

To train on the base configuration (use Human3.6M for training), run:

```
python src/learn_pose.py \
	--posedsac_only \
	--transfer -1 \
	--temp 1.8 \
	--gumbel \
	--entropy_beta_cam .01 \
	-lr 0.0005 \
	-lrs 10 \
	-ts 5 \
	--temp_gamma 0.9 \
	--train_iterations 10 \
	--valid_iterations 1 \
	--pose_hypotheses 200 \
	--layers_posedsac 1000 900 900 900 900 700 \
	--entropy_beta_pose 0.01 \
	--est_beta 0.02 \
	--exp_beta 1. \
	--body_lengths_mode 2 \
	--pose_batch_size 16
```

A more convenient way to specify the arguments is through the .vscode/launch.json, if the VSCode IDE is used.


### Pose estimation model evaluation

Download pretrained models from [SharePoint](https://ferhr-my.sharepoint.com/:f:/g/personal/kbartol_fer_hr/EkaiHg-8FuhDtHhL9_2vquwBdRB6JiscuEbv15tc7-HvuQ?e=PBSLl7) (password: _pretrained-3d-humans_).

```
python src/learn_pose.py \
	--posedsac_only \
	--transfer -1 \
	--temp 1.8 \
	--gumbel \
	--entropy_beta_cam .01 \
	-lr 0.0005 \
	-lrs 10 \
	-ts 5 \
	--temp_gamma 0.9 \
	--train_iterations 10 \
	--valid_iterations 1 \
	--pose_hypotheses 200 \
	--layers_posedsac 1000 900 900 900 900 700 \
	--entropy_beta_pose 0.01 \
	--est_beta 0.02 \
	--exp_beta 1. \
	--body_lengths_mode 2 \
	--pose_batch_size 16 \
	--test
```


### Pose estimation model (inference)

To run an inference on novel views, first use a 2D keypoint detector in all views and frames to generated 2D keypoint estimates. 

Once the poses are obtained, you can run `src/inference.py`.


### Relative camera pose estimation (inference)

To estimate relative camera poses on Human3.6M using the keypoint estimation on the test data, run:

```
python src/fundamental.py
```

The rotation and translation estimations are produced and stored in `est_Rs.npy` and `est_ts.npy`.

### Results

The results for base, intra, and inter configurations are:

| Base (H36M) | Intra (CMU) | Inter (CMU->H36M) |
| --- | --- | --- |
| 29.1 mm | 25.6 mm | 31.0 mm |


### Data and pretrained models

The data used for the above commands is in ./data/ folder. Note that, in this submission, we only 
include subject 1 (Human3.6M) for training, but it should be sufficient to reproduce
the original results.

### Acknowledgements

Parts of the source code were adapted from [cvlab-dresden/DSAC](https://github.com/cvlab-dresden/DSAC) and [karfly/learnable-triangulation-pytorch](https://github.com/karfly/learnable-triangulation-pytorch) and directly inspired by some of the following publications:

[1] [DSAC - Differentiable RANSAC for Camera Localization](https://arxiv.org/abs/1611.05705)

[2] [Learnable Triangulation of Human Pose](https://arxiv.org/abs/1905.05754)

[3] [Neural-Guided RANSAC: Learning Where to Sample Model Hypotheses](https://arxiv.org/abs/1905.04132)

[4] [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)

[5] [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712)
