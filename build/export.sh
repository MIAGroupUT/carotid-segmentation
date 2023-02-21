#!/usr/bin/env bash

case $1 in
  'grand-challenge')
    build/build.sh 'grand-challenge'
    docker save carotidsegmentation-grandchallenge | gzip -c > build/CarotidSegmentationGrandChallenge.tar.gz
    ;;
  *)
    build/build.sh
    docker save carotidsegmentation | gzip -c > build/CarotidSegmentation.tar.gz
    ;;
esac