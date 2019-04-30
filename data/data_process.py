#This is a script for data processing
import tensorflow as tf
import numpy as np
import cv2
from yolo_model import YOLO

import os
import sys
import json
from utils import *

YOLO_CLASSIFICATION_CLASSES_FILE = "./yolo_data/coco_classes.txt"
YOLO_MODEL = "./yolo_data/yolo.h5"

device_cf = tf.ConfigProto(device_count={'GPU': 1},
						   inter_op_parallelism_threads = 2,
						   intra_op_parallelism_threads = 2,
						   log_device_placement = False)
device_cf.gpu_options.per_process_gpu_memory_fraction = 0.1
device_cf.gpu_options.allow_growth = True

def get_yolo_model(path_to_model, obj_threshold=0.6, nms_threshold=0.5):
	return YOLO(path_to_model, obj_threshold, nms_threshold)

def process_image(img):
	image = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
	image = np.array(image, dtype='float32')
	image /= 255.
	image = np.expand_dims(image, axis=0)
	return image

def crop(img, box):
	# box: coordinate of box
	x, y, w, h = box
	top = max(0, np.floor(x + 0.5).astype(int))
	left = max(0, np.floor(y + 0.5).astype(int))
	right = min(img.shape[1], np.floor(x + w + 0.5).astype(int))
	bottom = min(img.shape[0], np.floor(y + h + 0.5).astype(int))
	sub_region = img[left:bottom, top:right, :]
	return sub_region

def detect_with_yolo(image, yolo, name, option=None):
	'''
	:param image: image
	:param yolo: YOLOv3 model
	:param name: name of the image
	:return: croped person
	'''
	if(option=='BBOX'):
		image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_CUBIC)
	pimage = process_image(image) # resize, normalize
	boxes, classes, scores = yolo.predict(pimage, image.shape)
	# if boxes is None or 0 not in classes:
	# yolo detect nothing or does not detect human
	if boxes is None or 0 not in classes:
		print('No detected item on image {}'.format(name))
		if option=='BBOX':
			bbox = [0, 0, 416, 416]
			return bbox, image
		else:
			return image
	max_area = 0
	for box, cls, score in zip(boxes, classes, scores):
		if cls != 0: # yolo does not detect human
			continue
		x, y, w, h = box
		if(w*h>max_area):
			max_box = box # Find the largest bounding max on the image

	img_person = crop(image, max_box)
	if option=='BBOX':
		return list(max_box), img_person
	else:
		
		return img_person

def process_deep_fashion(dataset_dir, output_dir):
	yolo = get_yolo_model(YOLO_MODEL)
	print("Yolo and gesture_model models are loading...")
	print("Processing target directory: ", dataset_dir)
	print("Save to directory: ", output_dir)
	if not os.path.exists(output_dir + 'annotation/'):
		os.makedirs(output_dir + 'annotation/')
	if not os.path.exists(output_dir + 'images/'):
		os.makedirs(output_dir + 'images/')

	file_output = open(output_dir + 'annotation/' + "image_anno.txt","a")
	for subdir, dirs, files in os.walk(dataset_dir):
		for file in files:
			if(file.endswith('.jpg') or file.endswith('.jpeg')):
				folder_id = subdir.split('/')[-1]
				# print(folder_id+'_'+file)
				img_path = os.path.join(subdir, file)
				image = cv2.imread(img_path,cv2.IMREAD_COLOR)
				img_person = detect_with_yolo(image, yolo, file, option='CROP')
				print("Detect person in image {}!".format(folder_id+'_'+file))
				save_path = output_dir + 'images/' + 'processed_' + folder_id+'_'+file
				file_output.write(save_path + '\n')
				cv2.imwrite(save_path, img_person)

def process_human_pose(dataset_dir, output_dir):
	yolo = get_yolo_model(YOLO_MODEL)
	print("Yolo and gesture_model models are loading...")
	print("Processing target directory: ", dataset_dir)
	print("Save to directory: ", output_dir)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	data = {}
	image_annos = []
	for subdir, dirs, files in os.walk(dataset_dir):
		for file in files:
			if(file.endswith('.jpg') or file.endswith('.jpeg')):
				image_anno = {}
				img_path = os.path.join(subdir, file)
				image = cv2.imread(img_path,cv2.IMREAD_COLOR)
				bbox, img_person = detect_with_yolo(image, yolo, file, option='BBOX')
				print("Detect person in image {}!".format(file))
				mask, pose_vector = get_body_vector(img_person)
				image_anno['mask'] = mask.tolist()
				image_anno['image_path'] = img_path
				image_anno['bbox'] = bbox
				image_anno['pose_vector'] = pose_vector.tolist()
				image_annos.append(image_anno)

	data['annotations'] = image_annos

	print("Save to JSON file: ", output_dir+'anno_bbox_pose.json')
	with open(output_dir+'anno_bbox_pose_test.json', 'w') as outfile:
		json.dump(data, outfile)

if __name__ == '__main__':
	deep_fashion_dir = "/home/lichenghui/deepfashion/img"
	deep_fashion_anno_dir = "/home/lichenghui/processed_deep_fashion_full/"
	human_pose_dir = '/home/lichenghui/mpii_human_pose/images'
	human_pose_anno_dir = '/home/lichenghui/mpii_human_pose/annotation/'


	if(len(sys.argv)>1):
		human_pose_dir = sys.argv[1:]
		human_pose_anno_dir = sys.argv[2:]

	test_dir = '/home/lichenghui/test_mpii/images'
	test_anno = '/home/lichenghui/test_mpii/annotation/'
	process_deep_fashion(deep_fashion_dir, deep_fashion_anno_dir)

	# process_human_pose(test_dir, test_anno)

