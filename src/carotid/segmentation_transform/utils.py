from typing import Dict, Any


class SegmentationTransform:
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    def __call__(self, sample):
        pass
