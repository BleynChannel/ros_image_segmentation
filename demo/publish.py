import rospy
from sensor_msgs.msg import Image
import ros_numpy
import numpy as np
import cv2

rospy.init_node('publish', anonymous=True)
rate = rospy.Rate(100)

publish = rospy.Publisher('/camera_101/image_raw', Image, queue_size=1)
publish_2 = rospy.Publisher('/camera_102/image_raw', Image, queue_size=1)

# from PIL import Image as PILImage

# image = PILImage.open('demo/cityscapes_demo.png')
# image = np.array(image)
# image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (960, 480), cv2.INTER_LINEAR)
# image_msg = ros_numpy.msgify(Image, image, encoding='bgr8')

# publish.publish(image_msg)
# publish_2.publish(image_msg)

cap = cv2.VideoCapture('demo/city_movie.mov') # 'rtsp://192.168.31.233:8554/cam1'

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

import signal

def stop_signal_handler(signal_number, stack_frame):
    exit(0)

signal.signal(signal.SIGINT, stop_signal_handler)
signal.signal(signal.SIGTERM, stop_signal_handler)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        image = np.array(frame)
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (960, 480), cv2.INTER_LINEAR)
        image_msg = ros_numpy.msgify(Image, image, encoding='bgr8')
        publish.publish(image_msg)
        publish_2.publish(image_msg)
        rate.sleep()
 
    # Break the loop
    else: 
        break
 
# When everything done, release the video capture object
cap.release()