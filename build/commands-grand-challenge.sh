#!/usr/bin/env bash

carotid pipeline_transform /input /opt/algorithm/models/heatmap_transform /opt/algorithm/models/contour_transform /output/tmp
python refactor_outputs.py

rm -r /output/tmp
echo "###### output ########"
ls /output
echo "###### output/images ########"
ls /output/images
echo "###### output/results.json ########"
cat /output/results.json
