import tensorflow as tf
import numpy as np
import cv2
import torch
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from dataloader import *
# All the constants
import config

DUMPED_MODEL = config.dumped_model #"model_10_final.pth.tar"
INTER_DIM = config.inter_dim #512
CATEGORIES = config.categories #20


device_cf = tf.ConfigProto(device_count={'GPU': 1},
                           inter_op_parallelism_threads = 2,
                           intra_op_parallelism_threads = 2,
                           log_device_placement = False)
device_cf.gpu_options.per_process_gpu_memory_fraction = 0.9
device_cf.gpu_options.allow_growth = True
e = TfPoseEstimator(get_graph_path('cmu'), tf_config=device_cf)

def load_model(path=None):
    if not path:
        return None
    return torch.load(path) #, map_location='cpu'

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def encode_body_vector(human, width, height):
    body_vector = np.zeros(shape=(2, 18))
    mask = np.zeros(shape=(18))
    # print("Human body body_parts")
    # print(human.body_parts.keys())
    # print("width, height", width, height)
    for i in range(18):
        if i not in human.body_parts.keys():
            # body_vector.append([0,0])
            body_vector[0, i] = 0
            body_vector[1, i] = 0
            mask[i] = 0
            continue
        body_part = human.body_parts[i]
        # print("key: {}, value: x-{}, y-{}".format(i, body_part.x, body_part.y))
        body_vector[0, i] = int(body_part.x * width + 0.5)
        body_vector[1, i] = int(body_part.y * height + 0.5)
        mask[i] = 1

    # for i in range(18):
    #     # convert to relative coordinate, centered at neck i=1
    #     if body_vector[i][0] or body_vector[i][1]:
    #         x = body_vector[i][0]
    #         y = body_vector[i][1]
    #         # rho, phi = cart2pol(x,y)
    #         body_vector[i] = [x, y]

    # body_vector = np.asarray(body_vector)
    # print(body_vector.shape)
    return mask, body_vector

def get_heatMap(img_dir):
    resize = '0x0'
    w, h = model_wh(resize)
    image = common.read_imgfile(img_dir, None, None)

    resize_out_ratio = 8.0
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

    heatMap = e.heatMat

    # tmp2 = e.pafMat.transpose((2, 0, 1))
    # odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    # even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
    heatMap = cv2.resize(heatMap, (112, 224), interpolation=cv2.INTER_CUBIC)
   

    # heapmap average to save some space
    heatMap = np.mean(heatMap, axis=2)
    # print(heatMap.shape)
    return heatMap

def get_body_vector(image):
    resize = '224x224'
    w, h = model_wh(resize)

    resize_out_ratio = 8.0
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

    mask = np.zeros(shape=(18,))
    if(len(humans)>0):
        # print("Pose estimator: {} humans".format(len(humans)))
        mask, body_vector = encode_body_vector(humans[0],w, h)
    else:
        print("NO HUMAN IN POSE ESTIMATION")
        body_vector = np.zeros(shape=(2, 18))
    # shape = (2, 18)
    # print(body_vector.shape)
    return mask, body_vector


def get_batch_body_vector(batch_image):
    # output: (N, 2, 18)
    resize = '224x224'
    w, h = model_wh(resize)
    # image = common.read_imgfile(img_path, None, None)

    body_vectors = np.zeros(shape=(batch_image.shape[0], 2, 18))
    masks = np.zeros(shape=(batch_image.shape[0], 18))
    for i in range(batch_image.shape[0]):
        image = batch_image[i]
        resize_out_ratio = 8.0
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

        if(len(humans)>0):
            mask[i], body_vectors[i] = encode_body_vector(humans[0],w, h)
        else:
            body_vectors[i] = np.zeros(shape=(2, len(human.body_parts)))
        # shape = (2, 18)
        # print(body_vector.shape)
    return body_vectors
