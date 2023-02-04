#!/usr/bin/env bash

./build.sh

docker save carotidsegmentation | gzip -c > CarotidSegmentation.tar.gz
