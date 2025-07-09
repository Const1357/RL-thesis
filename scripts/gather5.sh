#!/bin/bash

command="$1"

for i in {1..3}; do
    echo "Run #$i"
    eval "$command"
done

# run example: ./scripts/gather5 "./scripts/runner cartpole logits noise"