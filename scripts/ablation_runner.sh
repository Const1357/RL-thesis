#!/bin/bash

mod=$1
seed=$2

python3 main.py --common configs/pendulum_ablation/pendulum_ablation_common.yaml --config configs/pendulum_ablation/pendulum_ablation_${mod}.yaml -seed ${seed}