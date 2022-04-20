# Generalizable Human Pose Triangulation

## Introduction

This is the source code for the CVPR 2022 paper [Generalizable Human Pose Triangulation](https://arxiv.org/abs/2110.00280) (the most up-to-date preprint along with the Appendix will be available tomorrow, **Apr 21st**).
The repository will contain the instructions on installation and demos for training and testing
the presented model, as well as for estimating fundamental matrices. We will include
the neccessary data needed to run the demos. The materials should be complete by mid May.

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

## Work-In-Progress

We plan to completely prepare the source code with the pretrained models, demos, and videos by mid May. The to-do list consists of:

- [ ] Refactor the source code
- [ ] Complete documentation
- [ ] Pretrained pose estimation learning model
- [ ] Pretrained fundamental matrix estimation learning model
- [ ] Demo (main) scripts to run training, inference, and evaluation
- [ ] Video demonstration

## Usage

First download pretrained backbone: https://github.com/karfly/learnable-triangulation-pytorch#model-zoo,
and place it in ./models/pretrained/.

To install and prepare the environment, use docker:

```
docker build -t <image-name> .

docker run --rm --gpus all --name <container-name> -it \
	-v ${REPO_DIR}:/generalizable-triangulation \
	-v ${BASE_DATA_DIR}/:/data/  <image-name>
```


### Triangulation model training

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
	--pose_hypotheses 200 \\
	--layers_posedsac 1000 900 900 900 900 700 \\
	--entropy_beta_pose 0.01 \\
	--est_beta 0.02 \\
	--exp_beta 1. \\
	--body_lengths_mode 2 \\ 
	--pose_batch_size 16
```

A more convenient way to specify the arguments is through the .vscode/launch.json, if the VSCode IDE is used.


### Relative camera pose estimation

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
