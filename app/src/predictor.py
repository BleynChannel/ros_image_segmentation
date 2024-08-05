import logging
import numpy as np
import cv2

from .config import AppConfig
from .models import Model

def draw_segmentation_map(labels: np.ndarray[np.uint8], color_scheme: np.ndarray[np.uint8]):
    # Create 3 Numpy arrays containing zeros.
    # Later each pixel will be filled with respective red, green, and blue pixels
    # depending on the predicted class.

    red_map   = np.zeros_like(labels, dtype=np.uint8)
    green_map = np.zeros_like(labels, dtype=np.uint8)
    blue_map  = np.zeros_like(labels, dtype=np.uint8)

    for label_num in range(0, len(color_scheme)):
        index = labels == label_num

        R, G, B = color_scheme[label_num]

        red_map[index]   = R
        green_map[index] = G
        blue_map[index]  = B

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image, segmented_image):
    alpha = 1  # transparency for the original image
    beta  = 0.8  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)

    return image

class Predictor:
    def __init__(self, config: AppConfig):
        self.config = config 
        model_path = self.config.model_path
        device = self.config.device

        if config.model_type == 'paddleseg':
            from .models.paddleseg import PaddleSeg
            self.model: Model = PaddleSeg(model_path, device)
        elif config.model_type == 'autoseg':
            from .models.autoseg import AutoSeg
            self.model: Model = AutoSeg(model_path, device)

        logging.info(f'Loaded {self.config.model_type} model from {model_path}')

    def normalize(self, images: np.ndarray[np.uint8]) -> np.ndarray[np.float32]:
        # input -> shape: (batch or ?, height, width, channel), dtype: uint8, range: [0, 255]
        if images.ndim == 3:
            images = images[np.newaxis, ...]

        images = images.transpose(0, 3, 1, 2) # (batch, channel, height, width)
        images = images.astype(np.float32) / 255.0

        # output -> shape: (batch, channel, height, width), dtype: float32, range: [0, 1]
        return images

    def predict(self, inputs: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
        return self.model.predict(inputs)
    
    def __segmented_and_overlayed__(self, image: np.ndarray[np.uint8], labels: np.ndarray[np.uint8]) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:        
        # Get RGB segmentation map
        segmented_image = draw_segmentation_map(labels, self.model.color_scheme)
        overlayed_image = image_overlay(image, segmented_image)

        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)

        return segmented_image, overlayed_image

    def segmented_and_overlayed(self, images: np.ndarray[np.uint8], labels: np.ndarray[np.uint8]) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:
        # images shape: (batch, height, width, channels)
        # labels shape: (batch, height, width)

        if labels.ndim == 3:
            segmented_images, overlayed_images = [], []

            for image, label in zip(images, labels):
                segmented_image, overlayed_image = self.__segmented_and_overlayed__(image, label)

                segmented_images.append(segmented_image)
                overlayed_images.append(overlayed_image)
            
            return np.stack(segmented_images), np.stack(overlayed_images)
