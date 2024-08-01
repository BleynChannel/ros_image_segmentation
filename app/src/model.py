from time import time
import numpy as np
import cv2
import onnxruntime as ort

import rospy

labels = np.array([
	"background",
	"aeroplane",
	"bicycle",
	"bird",
	"boat",
	"bottle",
	"bus",
	"car",
	"cat",
	"chair",
	"cow",
	"dining table",
	"dog",
	"horse",
	"motorbike",
	"person",
	"potted plant",
	"sheep",
	"sofa",
	"train",
	"tv/monitor"
])

label_map = np.array([
	(0, 0, 255),  # background
	(128, 0, 0),  # aeroplane
	(0, 128, 0),  # bicycle
	(128, 128, 0),  # bird
	(0, 0, 128),  # boat
	(128, 0, 128),  # bottle
	(0, 128, 128),  # bus
	(128, 128, 128),  # car
	(64, 0, 0),  # cat
	(192, 0, 0),  # chair
	(64, 128, 0),  # cow
	(192, 128, 0),  # dining table
	(64, 0, 128),  # dog
	(192, 0, 128),  # horse
	(64, 128, 128),  # motorbike
	(192, 128, 128),  # person
	(0, 64, 0),  # potted plant
	(128, 64, 0),  # sheep
	(0, 192, 0),  # sofa
	(128, 192, 0),  # train
	(0, 64, 128),  # tv/monitor
])

class_map = dict(enumerate(labels))

def draw_segmentation_map(labels):
	# Create 3 Numpy arrays containing zeros.
	# Later each pixel will be filled with respective red, green, and blue pixels
	# depending on the predicted class.

	red_map   = np.zeros_like(labels, dtype=np.uint8)
	green_map = np.zeros_like(labels, dtype=np.uint8)
	blue_map  = np.zeros_like(labels, dtype=np.uint8)

	for label_num in range(0, len(label_map)):
		index = labels == label_num

		R, G, B = label_map[label_num]

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

# from transformers import MobileViTFeatureExtractor, MobileViTForSemanticSegmentation
# import torch

# feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/deeplabv3-mobilevit-xx-small")
# model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-xx-small")


class Model:
	def __init_onnx__(self, model_path):
		self.session = ort.InferenceSession(model_path)

		inputs = self.session.get_inputs()
		outputs = self.session.get_outputs()

		assert len(inputs) > 0
		assert len(outputs) > 0
		
		input = inputs[0]
		self.width, self.height = [512 if type(input) == str else input for input in input.shape[2:]]
		self.input_name = input.name
		
		# output = outputs[0]
		# self.output_name = output.name
	
	def __init_tflite__(self, model_path):
		pass

	def __init__(self, model_path) -> None:
		if model_path.endswith('.onnx'):
			self.backend = 'ONNX'
			self.__init_onnx__(model_path)
		elif model_path.endswith('.tflite'):
			self.backend = 'TFLITE'
			self.__init_tflite__(model_path)
		
		# self.running = False

	def run_inference(self, img_raw):
		# inputs = feature_extractor(images=image, return_tensors="pt")

		# outputs = model(**inputs)

		# logits are of shape (batch_size, num_labels, height, width)
		# predicted_mask = outputs.logits.argmax(1).squeeze(0)
		# return predicted_mask

		# feature_extractor = MobileViTFeatureExtractor.from_pretrained()
		# model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-xx-small")

		
		# model_inputs = feature_extractor(images=image, return_tensors="pt")
		# outputs = model(**model_inputs)
		# predicted_mask = outputs.logits.argmax(1).squeeze(0)
		# print(predicted_mask.shape)

		# self.running = True
		if self.backend == 'ONNX':
			return self.session.run(None, {self.input_name: img_raw})[0]
		elif self.backend == 'TFLITE':
			pass
		# self.running = False

	def run_with_benchmark(self, img_raw):
		start = time()
		outputs = self.run_inference(img_raw)
		inf_time = time() - start
		rospy.loginfo(f'Inference time: {inf_time} sec')
		return outputs

	def perform_inference(self, img_raw, change):
		# Pre-process image
		# image -> shape: (height, width, channels); size: (?, ?); type: uint8; value range: 0 - 255
		H, W = img_raw.shape[:2]
		image = cv2.resize(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), (self.width, self.height), cv2.INTER_LINEAR)
		#TODO: Преобразование делать свое для каждой модели
		if change:
			image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
		else:
			image = image.transpose(2, 1, 0).astype(np.float32) / 255.0

		image = image[np.newaxis, ...]
		# print(img_raw.shape, img_raw.dtype, np.max(img_raw))
		# image -> shape: (batch_size, channels, width, height); size: (512, 512); type: float32; value range: 0.0 - 1.0


		# # Model Inference
		outputs = self.run_with_benchmark(image)
		# outputs -> shape: (batch_size, channels, width, height); size: (32, 32); type: float32; value range: 0.0 - 21.0
		
		labels = np.asarray(outputs).argmax(1).squeeze(0).astype(np.uint8)
		# labels -> shape: (width, height); size: (32, 32); type: uint8; value range: 0 - 21

		# Get RGB segmentation map
		segmented_image = draw_segmentation_map(labels)

		# Change background to black
		seg_image_copy = segmented_image.copy()
		stack_image = seg_image_copy.reshape(-1, 3)
		mask = (stack_image == label_map[0]).all(axis=1).nonzero()[0]
		stack_image[mask] = (0, 0, 0)

		# Resize to original image size
		segmented_image = cv2.resize(segmented_image, (W, H), cv2.INTER_LINEAR)
		seg_image_copy = cv2.resize(seg_image_copy, (W, H), cv2.INTER_LINEAR)
		
		overlayed_image = image_overlay(img_raw, seg_image_copy)

		segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
		overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)

		return segmented_image, overlayed_image
	
	def __call__(self, image, change):
		return self.perform_inference(image, change)