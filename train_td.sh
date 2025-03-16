#!/bin/bash
# Script to set up and run TD learning for Ultimate Volleyball

# Activate the Conda environment
# Comment this line if you're activating the environment separately
source activate volleyball-td

# Register the TD trainer with ML-Agents
echo "Registering TD trainer with ML-Agents..."
python register_td_trainer.py

if [ $? -ne 0 ]; then
    echo "Failed to register TD trainer. Exiting."
    exit 1
fi

# Parse command line arguments
RUN_ID="volleyball_td"
TIME_SCALE=1.0
RESUME=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --run-id)
        RUN_ID="$2"
        shift
        shift
        ;;
        --time-scale)
        TIME_SCALE="$2"
        shift
        shift
        ;;
        --resume)
        RESUME=true
        shift
        ;;
        *)
        echo "Unknown argument: $1"
        shift
        ;;
    esac
done

# Build ML-Agents command
COMMAND="mlagents-learn config/Volleyball_TD.yaml --run-id=$RUN_ID --time-scale=$TIME_SCALE"

if [ "$RESUME" = true ]; then
    COMMAND="$COMMAND --resume"
fi

# Run the training
echo "Starting TD training with command:"
echo "$COMMAND"
$COMMAND

echo "Training complete!"