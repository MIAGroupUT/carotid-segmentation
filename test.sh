#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

# Generate random number to name the temporary output volume
VOLUME_SUFFIX=$RANDOM
MEM_LIMIT="4g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

# Create a volume to write outputs
docker volume create carotidsegmentation-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/tests/raw_dir/:/input/ \
        -v carotidsegmentation-output-$VOLUME_SUFFIX:/output/ \
        carotidsegmentation "--device cpu"

# Remove the volume containing the outputs
docker volume rm carotidsegmentation-output-$VOLUME_SUFFIX
