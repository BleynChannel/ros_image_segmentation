import argparse
import numpy as np
import signal
import sys

import rospy
from sensor_msgs.msg import Image

np.float = np.float32
import ros_numpy

from src.model import Model
from src.camera import Camera

is_stopping = False

def main(args):
	model = Model(model_path=args.model)

	rospy.init_node('test')

	def get_image(msg: Image):
		# image shape (height, width, channels)
		image = ros_numpy.numpify(msg)
		return model(image, args.c)

	cameras = []
	for camera_idx in range(1, args.count_cameras + 1):
		topic_name = f'/camera_10{camera_idx}'

		camera = Camera(topic_name, get_image)
		camera.start()
		cameras.append(camera)

	def signal_handler(sig, frame):
		global is_stopping

		if not is_stopping:
			is_stopping = True
			rospy.loginfo('Stopping inference...')

			for camera in cameras:
				camera.stop()
				camera.join()

		rospy.loginfo('Exiting...')
		sys.exit(0)

	signal.signal(signal.SIGINT, signal_handler)
	rospy.loginfo('Inference started')
	signal.pause()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('model', type=str)
	# parser.add_argument('config', type=str)
	parser.add_argument('--count_cameras', type=int, default=1)
	parser.add_argument('-c', action='store_true')
	# parser.add_argument('')
	
	args = parser.parse_args()
	main(args)