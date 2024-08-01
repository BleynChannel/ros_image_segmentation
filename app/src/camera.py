from threading import Thread
from sensor_msgs.msg import Image
import rospy
import ros_numpy

class Camera(Thread):
	def __init__(self, topic_name, callback):
		Thread.__init__(self)

		self.topic_name = topic_name
		self.raw_topic_name = f'{topic_name}/image_raw'
		self.callback = callback

		self.segmented_publisher = rospy.Publisher(f'{topic_name}/segmented_image', Image, queue_size=1)
		self.overlayed_publisher = rospy.Publisher(f'{topic_name}/overlayed_image', Image, queue_size=1)

		self.running = False

	def run(self):
		self.running = True
		while self.running and not rospy.is_shutdown():
			msg = rospy.wait_for_message(self.raw_topic_name, Image)
			segmented_image, overlayed_image = self.callback(msg)

			self.segmented_publisher.publish(ros_numpy.image.numpy_to_image(segmented_image, msg.encoding))
			self.overlayed_publisher.publish(ros_numpy.image.numpy_to_image(overlayed_image, msg.encoding))

	def stop(self):
		self.running = False