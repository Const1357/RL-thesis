#!/bin/bash

mod=$1
seed=$2

# If mod is empty or looks like a number (i.e. user passed seed as $1)
if [[ -z "$mod" || "$mod" =~ ^[0-9]+$ ]]; then
    seed="$1"
    config_file="configs/acrobot_ablation/acrobot_ablation_.yaml"
else
    config_file="configs/acrobot_ablation/acrobot_ablation_${mod}.yaml"
fi

# Run with or without seed
if [[ -z "$seed" ]]; then
    python3 main.py --common configs/acrobot_ablation/acrobot_ablation_common.yaml --config "$config_file"
else
    python3 main.py --common configs/acrobot_ablation/acrobot_ablation_common.yaml --config "$config_file" --seed "$seed"
fi

# python3 main.py --common configs/acrobot_ablation/acrobot_ablation_common.yaml --config configs/acrobot_ablation/acrobot_ablation_${mod}.yaml -seed ${seed}