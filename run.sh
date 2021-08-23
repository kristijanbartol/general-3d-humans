#!/bin/sh

if [[ "$(whoami)" == "kristijan" ]]; then
			REPO_DIR=/media/kristijan/kristijan-hdd-ex/poseDSAC/
			BASE_DATA_DIR=/media/kristijan/kristijan-hdd-ex/datasets/
				else
			REPO_DIR=/home/dbojanic/pose/poseDSAC/
			BASE_DATA_DIR=/home/dbojanic/datasets/
fi

docker run --rm --gpus all --name kbartol-posedsac -it \
	-v ${REPO_DIR}:/poseDSAC \
	-v ${BASE_DATA_DIR}/human36m/:/data/human36m/ \
	-v /media/kristijan/kristijan-hdd-ex/panoptic-toolbox/scripts/:/data/cmupanoptic/ kbartol-posedsac

