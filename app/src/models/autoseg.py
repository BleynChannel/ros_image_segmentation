import numpy as np
import logging

from .model import Model

class AutoSeg(Model):
    def __init__(self, model_path: str, device = 'cpu'):
        super().__init__(model_path, device)

    def predict(self, inputs: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
        if inputs.shape[0] > 1:
            logging.warning("Model is not supported in batch mode. Using single prediction mode instead.")

        return super().predict(inputs)
