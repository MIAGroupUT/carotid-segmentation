from carotid.utils.transforms import LoadCSVd
from monai.transforms import Transform
from os import path, listdir, makedirs
from typing import Dict, Set, List, Any
from .template import Serializer


side_list = ["left", "right"]


class CenterlineSerializer(Serializer):
    def get_transforms(self) -> List[Transform]:
        transforms = [LoadCSVd(keys=["left_centerline", "right_centerline"], sep="\t")]
        return transforms

    def find_participant_set(self) -> Set[str]:
        return {
            participant_id
            for participant_id in listdir(self.parameters["dir"])
            if path.exists(
                path.join(
                    self.parameters["dir"], participant_id, "centerline_transform"
                )
            )
        }

    def add_path(self, sample_list: List[Dict[str, str]]):
        for sample in sample_list:
            centerline_dir = path.join(
                self.parameters["dir"], sample["participant_id"], "centerline_transform"
            )
            for side in side_list:
                sample[f"{side}_centerline"] = path.join(
                    centerline_dir, f"{side}_centerline.tsv"
                )

    def write(self, sample: Dict[str, Any]):
        output_dir = path.join(
            self.parameters["dir"], sample["participant_id"], "centerline_transform"
        )
        makedirs(output_dir, exist_ok=True)

        for side in side_list:
            sample[f"{side}_centerline"].to_csv(
                path.join(output_dir, f"{side}_centerline.tsv"), sep="\t", index=False
            )
