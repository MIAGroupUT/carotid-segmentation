from carotid.utils import build_dataset
from monai.data import ITKWriter
import numpy as np
from os import makedirs, path, listdir

input_dir = "/input"
output_dir = "/output"
makedirs(path.join(output_dir, "images"), exist_ok=True)
print(listdir(output_dir))

writer = ITKWriter(output_dtype="uint8")
dataset = build_dataset(raw_dir=input_dir, segmentation_dir=output_dir)
for sample in dataset:
    participant_id = sample["participant_id"]
    wall_segmentation = 0
    for side in ["left", "right"]:
        wall_segmentation += sample[f"{side}_wall_segmentation"]
    wall_segmentation = np.clip(wall_segmentation, 0, 1)
    writer.set_data_array(wall_segmentation)
    writer.set_metadata({"affine": sample["image"].affine})
    writer.write(path.join(output_dir, "images", f"{participant_id}.mha"))
