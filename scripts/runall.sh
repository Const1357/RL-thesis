#!/bin/bash

# envs=("cartpole" "pendulum")
# types=("logits" "GNN" "GNN_K" "GNN_N")
# mods=("" "noise" "entropy" "noise_entropy")

# for env in "${envs[@]}"; do
#     for type in "${types[@]}"; do
#         for mod in "${mods[@]}"; do
#             ./scripts/gather5.sh "./scripts/runner.sh $env $type $mod"
#         done
#     done
# done

# Manual unrolled (explicit) version because of differences in configurations (GNN_N does not have noise, pong does not have GNN, GNN_K etc.)

# ./scripts/gather5.sh "./scripts/runner.sh cartpole logits"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole logits entropy"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole logits noise"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole logits noise_entropy"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN entropy"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN noise"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN noise_entropy"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN_K"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN_K entropy"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN_K noise"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN_K noise_entropy"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN_N"
# ./scripts/gather5.sh "./scripts/runner.sh cartpole GNN_N entropy"

# ./scripts/gather5.sh "./scripts/runner.sh pendulum logits"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum logits entropy"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum logits noise"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum logits noise_entropy"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum GNN"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum GNN entropy"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum GNN noise"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum GNN noise_entropy"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum GNN_K"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum GNN_K entropy"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum GNN_K noise"
# ./scripts/gather5.sh "./scripts/runner.sh pendulum GNN_K noise_entropy"
./scripts/gather3.sh "./scripts/runner.sh pendulum GNN_N"
# ./scripts/gather3.sh "./scripts/runner.sh pendulum GNN_N entropy"

# ./scripts/gather3.sh "./scripts/runner.sh pong logits entropy"
# ./scripts/gather3.sh "./scripts/runner.sh pong GNN_N entropy"
