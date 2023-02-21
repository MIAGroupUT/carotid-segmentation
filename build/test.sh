#!/usr/bin/env bash

SCRIPTPATH="$( cd $(dirname "$(dirname "$0")") ; pwd -P )"

# Generate random number to name the temporary output volume
VOLUME_SUFFIX=$RANDOM
MEM_LIMIT="15g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

TESTPATH="$SCRIPTPATH/tests/tmp"

if [ -d $TESTPATH ]; then
  rm -r $TESTPATH
fi

mkdir $TESTPATH

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
        -v $SCRIPTPATH/tests/raw_dir/:/input/ \
        -v $TESTPATH:/output/ \
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

# rm -r $SCRIPTPATH/tests/tmp
