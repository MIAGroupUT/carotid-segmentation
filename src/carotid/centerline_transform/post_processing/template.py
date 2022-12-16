import abc
from typing import Dict
import numpy as np


class CenterlineExtractor:
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(
        self, sample: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        pass
