from carotid.utils import build_dataset, read_json, write_json
from monai.data import ITKWriter
import numpy as np
from os import makedirs, path
import argparse

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('input_dir', type=str)
parser.add_argument('output_dir', type=str)

args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
tmp_dir = path.join(output_dir, "tmp")
makedirs(path.join(output_dir, "images"), exist_ok=True)

writer = ITKWriter(output_dtype="uint8")
dataset = build_dataset(raw_dir=input_dir, segmentation_dir=tmp_dir)
pipeline_dict = read_json(path.join(tmp_dir, "pipeline_parameters.json"))
write_json(pipeline_dict, path.join(output_dir, "results.json"))
for sample in dataset:
    participant_id = sample["participant_id"]
    wall_segmentation = 0
    for side in ["left", "right"]:
        wall_segmentation += sample[f"{side}_segmentation"][1:]
    wall_segmentation = np.clip(wall_segmentation, 0, 1)
    writer.set_data_array(wall_segmentation)
    writer.set_metadata({"affine": sample["image"].affine})
    writer.write(path.join(output_dir, "images", f"{participant_id}.mha"))

