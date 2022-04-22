#!/bin/bash

USERNAME=${1-$USER}

docker build -t $USERNAME-general-3d-humans ./docker 
