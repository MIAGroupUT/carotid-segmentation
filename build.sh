#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

case $1 in
  'test')
    docker build -t carotidsegmentation-test -f debian.docker --build-arg MODELPATH="models/" $SCRIPTPATH
    ;;
  'test-grand-challenge')
    docker build -t carotidsegmentation-test-grandchallenge -f grand-challenge.docker --build-arg MODELPATH="tests/models/" $SCRIPTPATH
    ;;
  'grand-challenge')
    docker build -t carotidsegmentation-grandchallenge -f grand-challenge.docker --build-arg MODELPATH="tests/models/" $SCRIPTPATH
    ;;
  *)
    docker build -t carotidsegmentation -f debian.docker --build-arg MODELPATH="models/" $SCRIPTPATH
    ;;
esac
