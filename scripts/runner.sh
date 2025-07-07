#!/bin/bash

env=$1
type=$2
mod=$3

python3 main.py --common configs/$env/${env}_common.yaml --config configs/$env/${env}_${type}_${mod}.yaml

# example run:
# ./scripts/runner.sh cartpole logits
# ./scripts/runner.sh cartpole logits noise
# ./scripts/runner.sh cartpole logits entropy
# ./scripts/runner.sh cartpole logits noise_entropy