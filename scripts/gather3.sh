#!/bin/bash

command="$1"

for i in {1..3}; do
    echo "Run #$i"
    eval "$command"
done

# run example: ./scripts/gather3 "./scripts/runner pong logits entropy"