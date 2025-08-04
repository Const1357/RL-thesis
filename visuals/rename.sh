#!/bin/bash

# Set working directory to the location of the script
cd "$(dirname "$0")"

# Create the output directory
mkdir -p renamed

# Loop through all files in visuals/ (but not in subdirectories)
for f in *; do
    # Skip if not a file
    [[ -f "$f" ]] || continue

    # Identify environment
    if [[ "$f" == *acrobot* ]]; then
        env_name="acrobot"
    elif [[ "$f" == *pong* ]]; then
        env_name="pong"
    elif [[ "$f" == *pendulum* ]]; then
        env_name="pendulum"
    else
        echo "Skipping: $f (no valid environment)"
        continue
    fi

    # Build modification code
    mod=""
    [[ "$f" == *alignment* ]] && mod+="A"
    [[ "$f" == *penalty* ]] && mod+="P"
    [[ "$f" == *margin* ]] && mod+="M"
    [[ -z "$mod" ]] && mod="N"

    # Extract extension
    ext="${f##*.}"

    # New filename
    new_name="${env_name}_${mod}.${ext}"

    # Copy to renamed/ folder
    cp "$f" "renamed/$new_name"
    echo "Renamed $f -> renamed/$new_name"
done