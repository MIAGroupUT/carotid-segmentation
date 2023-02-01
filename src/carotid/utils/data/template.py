import abc
from typing import Dict, Any, List, Set
from monai.transforms import Transform


class Serializer:
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    @abc.abstractmethod
    def get_transforms(self) -> List[Transform]:
        pass

    @abc.abstractmethod
    def find_participant_set(self) -> Set[str]:
        pass

    @abc.abstractmethod
    def add_path(self, sample_list: List[Dict[str, str]]):
        pass

    @abc.abstractmethod
    def write(self, sample: Dict[str, Any]):
        pass
