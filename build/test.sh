#!/usr/bin/env bash

SCRIPTPATH="$( cd $(dirname "$(dirname "$0")") ; pwd -P )"

# Generate random number to name the temporary output volume
VOLUME_SUFFIX=$RANDOM
MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

TESTPATH="$SCRIPTPATH/build/tests"

case $1 in
  'grand-challenge')
    build/build.sh "test-grand-challenge"
    docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $TESTPATH/input/:/input/ \
        -v $TESTPATH/output/:/output/ \
        --platform="linux/amd64" \
        carotidsegmentation-test-grandchallenge
    ;;
  *)
    build/build.sh "test"
    docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/tests/raw_dir/:/input/ \
        -v $TESTPATH:/output/ \
        carotidsegmentation-test
    ;;
esac

rm -r $TESTPATH/output/tmp
