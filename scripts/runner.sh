#!/bin/bash

env=$1
type=$2
mod=$3
seed=$4

# Construct the config path
if [[ -z "$mod" || "$mod" =~ ^[0-9]+$ ]]; then
    # If mod is empty or a number (i.e. seed was passed as $3), treat $3 as seed
    seed="$3"
    config_file="configs/$env/${env}_${type}_.yaml"
else
    config_file="configs/$env/${env}_${type}_${mod}.yaml"
fi

# Run the command with or without seed
if [[ -z "$seed" ]]; then
    python3 main.py --common "configs/$env/${env}_common.yaml" --config "$config_file"
else
    python3 main.py --common "configs/$env/${env}_common.yaml" --config "$config_file" --seed "$seed"
fi

# python3 main.py --common configs/$env/${env}_common.yaml --config configs/$env/${env}_${type}_${mod}.yaml --seed ${seed}

# example run:
# ./scripts/runner.sh cartpole logits
# ./scripts/runner.sh cartpole logits noise
# ./scripts/runner.sh cartpole logits entropy
# ./scripts/runner.sh cartpole logits noise_entropy

# example run with seed:
# ./scripts/runner.sh cartpole logits noise_entropy 1234
