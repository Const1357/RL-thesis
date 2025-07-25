#!/bin/bash

command="$1"

seeds=(4792 8870 5570) # from RNG

for i in {0..2}; do
    echo "Run #$((i + 1)) - seed=${seeds[$i]}"
    eval "$command ${seeds[$i]}"
done

# run example: ./scripts/gather3 "./scripts/runner pong logits entropy 1234"