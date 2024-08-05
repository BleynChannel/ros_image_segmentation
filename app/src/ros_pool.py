import logging
import time
import rospy
import numpy as np
np.float = np.float32
import ros_numpy
import threading

from sensor_msgs.msg import Image

from .config import AppConfig
from .predictor import Predictor
from .camera import Camera

class RosPool:
    def log_info(self, msg):
        rospy.loginfo(msg)
        logging.info(msg)

    def __predict_and_visualize__(self, images: np.ndarray[np.uint8]) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:
        # Normalize
        images_normalized = self.predictor.normalize(images)
        
        # Predict
        if self.config.benchmark:
            start = time.time()
            labels = self.predictor.predict(images_normalized)
            duration = time.time() - start

            self.log_info(f'Predicted in {duration} seconds')
        else:
            labels = self.predictor.predict(images_normalized)

        # Segmented and overlayed
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        
        segmented_images, overlayed_images = self.predictor.segmented_and_overlayed(images, labels)
        return segmented_images, overlayed_images

    def __init__(self, config: AppConfig, predictor: Predictor):
        self.config = config
        self.predictor = predictor

        rospy.init_node(self.config.node_name)
        self.log_info(f'ROS node {self.config.node_name} started')

        # Initialize
        self.log_info(f'Initialized ROS pool...')

        # Initialize cameras
        if self.config.predict_mode == 'single':
            def camera_callback(camera: Camera, msg: Image):
                # input shape (height, width, channel)
                image = ros_numpy.numpify(msg)
                image = image[np.newaxis, ...] # (batch, height, width, channel)

                # Predict
                segmented_image, overlayed_image = self.__predict_and_visualize__(image)

                # Publish
                camera.publish(segmented_image.squeeze(axis=0), overlayed_image.squeeze(axis=0), msg.encoding)

        elif self.config.predict_mode == 'batch':
            self.lock = threading.Lock()

            def camera_callback(camera: Camera, msg: Image):
                # Append msg to camera buffer
                self.cameras[np.where(self.cameras[:, 0] == camera)[0], 1] = msg

                with self.lock:
                    if np.all(self.cameras[:, 1] != None):
                        # input shape (height, width, channel)
                        images = np.stack([ros_numpy.numpify(msg) for msg in self.cameras[:, 1]]) # (batch, height, width, channel)

                        # Predict
                        segmented_images, overlayed_images = self.__predict_and_visualize__(images)

                        # Publish
                        for camera, segmented_image, overlayed_image in zip(self.cameras[:, 0], segmented_images, overlayed_images):
                            camera.publish(segmented_image, overlayed_image, msg.encoding)

                        # Clear buffer
                        self.cameras[:, 1] = None
        else:
            raise ValueError('Invalid predict_mode')

        if self.config.predict_mode == 'single':
            self.cameras = np.array([Camera(camera_config, camera_callback) for camera_config in self.config.cameras])
        elif self.config.predict_mode == 'batch':
            self.cameras = np.array([(Camera(camera_config, camera_callback), None) for camera_config in self.config.cameras])

        self.log_info(f'ROS pool initialized')

    def run(self):
        if self.config.predict_mode == 'single':
            cameras = self.cameras
        elif self.config.predict_mode == 'batch':
            cameras = self.cameras[:, 0]

        self.log_info('Starting ROS pool...')
        
        for camera in cameras:
            camera.start()

        self.log_info('ROS pool started')

        for camera in cameras:
            camera.join()

        self.log_info('ROS pool stopped')

    def stop(self):
        if self.config.predict_mode == 'single':
            cameras = self.cameras
        elif self.config.predict_mode == 'batch':
            cameras = self.cameras[:, 0]

        for camera in cameras:
            camera.stop()
