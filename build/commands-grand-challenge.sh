#!/usr/bin/env bash

carotid transform pipeline /input /opt/algorithm/models/heatmap_transform /opt/algorithm/models/contour_transform /output/tmp
python refactor_outputs.py /input /output
