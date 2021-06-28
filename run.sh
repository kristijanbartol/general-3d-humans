#!/bin/sh

if [[ "$(whoami)" == "kristijan" ]]; then
			REPO_DIR=/media/kristijan/kristijan-hdd-ex/poseDSAC/
				else
			REPO_DIR=/home/dbojanic/pose/poseDSAC/
fi

echo ${REPO_DIR}

docker run --rm --gpus all --name kbartol-posedsac -it \
	-v ${REPO_DIR}:/poseDSAC \
	-v /media/kristijan/kristijan-hdd-ex/datasets/human36m/:/data/human36m kbartol-posedsac
