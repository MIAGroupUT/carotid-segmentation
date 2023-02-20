from carotid.utils import build_dataset
from monai.data import ITKWriter
import numpy as np
from os import makedirs, path
import json

input_dir = "/input"
output_dir = "/output"
tmp_dir = path.join(output_dir, "tmp")
makedirs(path.join(output_dir, "images"), exist_ok=True)

results_dict = {
    "outputs": [],
    "inputs": [],
    "error_messages": [],
}

writer = ITKWriter(output_dtype="uint8")
dataset = build_dataset(raw_dir=input_dir, segmentation_dir=tmp_dir)
for sample in dataset:
    participant_id = sample["participant_id"]
    results_dict["inputs"].append(
        dict(type="metaio_image", filename=f"{participant_id}.mha")
    )
    results_dict["outputs"].append(
        dict(type="metaio_image", filename=f"{participant_id}.mha")
    )
    wall_segmentation = 0
    for side in ["left", "right"]:
        wall_segmentation += sample[f"{side}_segmentation"][1]
    wall_segmentation = np.clip(wall_segmentation, 0, 1)
    writer.set_data_array(wall_segmentation)
    writer.set_metadata({"affine": sample["image"].affine})
    writer.write(path.join(output_dir, "images", f"{participant_id}.mha"))

with open(path.join(output_dir, "results.json"), "w") as f:
    json.dump(results_dict, f)
