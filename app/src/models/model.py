import numpy as np

class Model:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.color_scheme = np.empty((0, 3), dtype=np.uint8)

    def predict(self, inputs: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
        pass
