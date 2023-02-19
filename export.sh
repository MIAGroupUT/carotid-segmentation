#!/usr/bin/env bash

case $1 in
  'grand-challenge')
    ./build.sh 'grand-challenge'
    docker save carotidsegmentation-grandchallenge | gzip -c > CarotidSegmentationGrandChallenge.tar.gz
    ;;
  *)
    ./build.sh
    docker save carotidsegmentation | gzip -c > CarotidSegmentation.tar.gz
    ;;
esac