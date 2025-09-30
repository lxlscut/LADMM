#!/bin/bash

# Specify the file to update
FILE_PATH="Trento.py"

# Define the list of random seeds
SEEDS=(5734 6265 466 5578 8322)

# Set the output directory
OUTPUT_DIR="result/trento3"

DEVICE="cuda:1"

# Iterate over each seed
for SEED in "${SEEDS[@]}"; do
    # Use sed to replace the seed in the script
    sed -i "s/setup_seed([0-9]\+)/setup_seed($SEED)/" $FILE_PATH

    # Run the script and save the output to a file
    python3 $FILE_PATH --device="$DEVICE" > "$OUTPUT_DIR/result_$SEED.txt"

    echo "Run completed for seed $SEED, result saved to $OUTPUT_DIR/result_$SEED.txt"
done
