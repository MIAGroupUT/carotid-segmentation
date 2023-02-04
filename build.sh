#!/bin/sh

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
MODELPATH="${SCRIPTPATH}/models"

if ! [[ -d MODELPATH ]]
then
  echo "Using test models to build the image..."
  cp -r "$SCRIPTPATH/tests/models" $MODELPATH
fi


docker build -t carotidsegmentation "$SCRIPTPATH"
