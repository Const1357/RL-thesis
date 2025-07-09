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

./scripts/gather5.sh "./scripts/runner.sh pong logits entropy"
./scripts/gather5.sh "./scripts/runner.sh pong GNN_N entropy"