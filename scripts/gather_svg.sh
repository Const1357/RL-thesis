#!/bin/bash

src_dir=".."
dst_dir="svg_collected"              # destination folder

mkdir -p "$dst_dir"

find "$src_dir" -type f -iname "*.svg" | while read -r file; do
    base=$(basename "$file")
    cp "$file" "$dst_dir/$base"

done

echo "All .svg files copied to $dst_dir"
