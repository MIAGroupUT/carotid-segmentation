#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
MODELPATH="models/"

case $1 in
  'test')
    MODELPATH="tests/models/"
    docker build -t carotidsegmentation-test -f debian.docker --build-arg MODELPATH=$MODELPATH $SCRIPTPATH
    ;;
  'test-grand-challenge')
    MODELPATH="tests/models/"
    docker build -t carotidsegmentation-test-grandchallenge -f grand-challenge.docker --build-arg MODELPATH=$MODELPATH $SCRIPTPATH
    ;;
  'grand-challenge')
    docker build -t carotidsegmentation-grandchallenge -f grand-challenge.docker --build-arg MODELPATH=$MODELPATH $SCRIPTPATH
    ;;
  *)
    docker build -t carotidsegmentation -f debian.docker --build-arg MODELPATH=$MODELPATH $SCRIPTPATH
    ;;
esac
