#!/bin/bash

# Specify the file to update
FILE_PATH="Houston.py"


# Define the list of random seeds
SEEDS=(7270 860 5390 5191 5734 6265 466 4426 5578 8322)

# Set the output directory
OUTPUT_DIR="result/houston_2025"

# Create the output directory if it does not exist
mkdir -p $OUTPUT_DIR

# Iterate over each seed
for SEED in "${SEEDS[@]}"; do
    # Use sed to replace the seed in the script
    sed -i "s/setup_seed([0-9]\+)/setup_seed($SEED)/" $FILE_PATH

    # Run the script and save the output to a file
    python3 $FILE_PATH > "$OUTPUT_DIR/result_$SEED.txt"

    echo "Run completed for seed $SEED, result saved to $OUTPUT_DIR/result_$SEED.txt"
done


#!/bin/bash

## Define the target script path
#FILE_PATH="Houston.py"
#
## Define the list of random seeds
#SEEDS=(7270 860 5390 5191 5734 6265 466 4426 5578 8322)
#
## Sweep lambda from 0.01 to 0.20 with a step of 0.01
#for lamda in $(seq 0.01 0.01 0.2); do
#    # Create the output directory
#    OUTPUT_DIR="result/houston_lamda_${lamda}"
#    mkdir -p "$OUTPUT_DIR"
#
#    # For each lambda value, run ten different seeds
#    for SEED in "${SEEDS[@]}"; do
#
#        sed -i "s/setup_seed([0-9]\+)/setup_seed($SEED)/" $FILE_PATH
#
#        # Pass lambda and seed through arguments
#        python3 $FILE_PATH --lamda "${lamda}"> "$OUTPUT_DIR/result_$SEED.txt"
#
#        echo "Run completed for lamda $lamda, seed $SEED, result saved to $OUTPUT_DIR/result_$SEED.txt"
#    done
#done
