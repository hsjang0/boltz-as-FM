#!/bin/bash

dir_name="$1"

if [ -z "$dir_name" ]; then
    echo "Usage: $0 <dir_name>"
    exit 1
fi


for dir in "${dir_name}"/mol_boltz_configs_*/; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        boltz predict "$dir" \
            --write_embeddings \
            --devices 8 \
            --num_workers 8 \
            --out_dir "$dir_name" \
            --override &
    else
        echo "Warning: $dir does not exist."
    fi
done
