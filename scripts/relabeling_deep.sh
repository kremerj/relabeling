#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 screen -dmL python experiment.py --experiment deep --dataset baidu --jobs 1
