#!/usr/bin/env bash

echo $0
echo $@
carotid pipeline_transform -h

echo "###### DEBUG #######"
ls /input
ls /output
ls /opt/algorithm/models

carotid pipeline_transform /input /opt/algorithm/models/heatmap_transform /opt/algorithm/models/contour_transform /output
python refactor_outputs.py