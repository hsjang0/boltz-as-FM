#!/bin/bash

dir_name="$1"

if [ -z "$dir_name" ]; then
    echo "Usage: $0 <dir_name>"
    exit 1
fi


for dir in "${dir_name}"/*_yaml_files/; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        boltz predict "$dir" \
            --write_embeddings \
            --devices 8 \
            --num_workers 8 \
            --cache boltz_for_ADMET \  
            --out_dir "$dir_name" \
            --override &
    else
        echo "Warning: $dir does not exist."
    fi
done
