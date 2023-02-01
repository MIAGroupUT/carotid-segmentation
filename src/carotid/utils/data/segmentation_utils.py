from typing import List, Dict, Any, Set
from monai.transforms import Transform

from .template import Serializer


class SegmentationSerializer(Serializer):
    def find_participant_set(self) -> Set[str]:
        pass

    def add_path(self, sample_list: List[Dict[str, str]]):
        pass

    def write(self, sample: Dict[str, Any]):
        pass

    def get_transforms(self) -> List[Transform]:
        pass
