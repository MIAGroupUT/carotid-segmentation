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
makedirs(path.join(output_dir, "images", "carotid-segmentation"), exist_ok=True)

writer = ITKWriter(output_dtype="uint8")
dataset = build_dataset(raw_dir=input_dir, segmentation_dir=tmp_dir)
pipeline_dict = read_json(path.join(tmp_dir, "parameters.json"))
write_json(pipeline_dict, path.join(output_dir, "results.json"))
for sample in dataset:
    participant_id = sample["participant_id"]
    segmentation_np = np.zeros_like(sample["image"])
    for side_idx, side in enumerate(["left", "right"]):
        for channel_idx in range(3, -1, -1):
            mask_np = sample[f"{side}_segmentation"][channel_idx]
            if channel_idx == 0:  # internal lumen
                value = 1 + 2 * side_idx
            elif channel_idx == 1:  # internal wall
                value = 5 + 2 * side_idx
            elif channel_idx == 2:  # external lumen
                value = 2 + 2 * side_idx
            else:  # external wall
                value = 6 + 2 * side_idx

            segmentation_np[0][mask_np == 1] = value

    writer.set_data_array(segmentation_np)
    writer.set_metadata({"affine": sample["image"].affine})
    writer.write(path.join(output_dir, "images", "carotid-segmentation", f"{participant_id}.mha"))

