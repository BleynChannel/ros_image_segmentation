from threading import Thread
from sensor_msgs.msg import Image
import rospy
import ros_numpy
import logging

from .config import CameraConfig

class Camera(Thread):
    def log_info(self, msg):
        rospy.loginfo(msg)
        logging.info(msg)

    def __init__(self, config: CameraConfig, callback):
        Thread.__init__(self)

        self.config = config
        self.callback = callback
        # self.rate = rospy.Rate(10)

        self.segmented_publisher = rospy.Publisher(self.config.segmented_image_topic, Image, queue_size=1)
        self.overlayed_publisher = rospy.Publisher(self.config.overlayed_image_topic, Image, queue_size=1)

        self.running = False

    def run(self):
        self.running = True
        self.log_info(f"Camera '{self.config.name}' started")
        while self.running and not rospy.is_shutdown():
            msg = rospy.wait_for_message(self.config.raw_topic, Image)
            self.callback(self, msg)
            # self.rate.sleep()

    def publish(self, segmented_image, overlayed_image, encoding = 'bgr8'):
        self.segmented_publisher.publish(ros_numpy.image.numpy_to_image(segmented_image, encoding))
        self.overlayed_publisher.publish(ros_numpy.image.numpy_to_image(overlayed_image, encoding))

    def stop(self):
        self.running = False
