#!/bin/bash

command="$1"

seeds=(4792 8870 5570 1760 610) # from RNG

for i in {0..4}; do
    echo "Run #$((i + 1)) - seed=${seeds[$i]}"
    eval "$command ${seeds[$i]}"
done

# run example: ./scripts/gather5 "./scripts/runner cartpole logits noise"