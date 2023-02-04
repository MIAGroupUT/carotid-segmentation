#!/bin/sh

./build.sh

docker save carotidsegmentation | gzip -c > CarotidSegmentation.tar.gz
