#!/usr/bin/env bash

SCRIPTPATH="$( cd $(dirname "$(dirname "$0")") ; pwd -P )"
echo $SCRIPTPATH

case $1 in
  'test')
    docker build -t carotidsegmentation-test -f build/debian.docker --build-arg MODELPATH="models/" $SCRIPTPATH
    ;;
  'test-grand-challenge')
    docker build -t carotidsegmentation-test-grandchallenge -f build/grand-challenge.docker --build-arg MODELPATH="tests/models/" $SCRIPTPATH
    ;;
  'grand-challenge')
    docker build -t carotidsegmentation-grandchallenge -f build/grand-challenge.docker --build-arg MODELPATH="tests/models/" $SCRIPTPATH
    ;;
  *)
    docker build -t carotidsegmentation -f build/debian.docker --build-arg MODELPATH="models/" $SCRIPTPATH
    ;;
esac
