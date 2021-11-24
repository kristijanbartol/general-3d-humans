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

To run the triangulation demo, use demo.py:

```
python demo.py --triangulation
```

To run the camera pose estimation demo, use demo.py:

```
python3 demo.py --camera_pose
```

The demos use data provided in ./data/ folder, as well as pretrained
models provided in ./models/ folder, by default.
