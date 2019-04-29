import torch
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# All the constants

DUMPED_MODEL = "model_10_final.pth.tar"
INTER_DIM = 512
CATEGORIES = 20


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
    return torch.load(path)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def encode_body_vector(human, width, height):
    body_vector = np.zeros(shape=(2, len(human.body_parts)))
    body_vector = []
    for i in range(18):
        if i not in human.body_parts.keys():
            body_vector.append([0,0])
            continue
        body_part = human.body_parts[i]
        body_vector.append([int(body_part.x * width + 0.5), int(body_part.y * height + 0.5)])
    for i in range(18):
        # convert to relative coordinate, centered at neck i=1
        if body_vector[i][0] or body_vector[i][1]:
            x = body_vector[i][0]
            y = body_vector[i][1]
            # rho, phi = cart2pol(x,y)
            body_vector[i] = [x, y]

    body_vector = np.asarray(body_vector)
    # print(body_vector.shape)
    return body_vector

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
    resize = '0x0'
    w, h = model_wh(resize)

    resize_out_ratio = 8.0
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

    if(len(humans)>0):
        body_vector = encode_body_vector(humans[0],w, h)
    else:
        body_vector = np.zeros(shape=(2, len(human.body_parts)))
    # shape = (2, 18)
    # print(body_vector.shape)
    return body_vector


def get_batch_body_vector(batch_image):
	# output: (N, 2, 18)
    resize = '0x0'
    w, h = model_wh(resize)
    # image = common.read_imgfile(img_path, None, None)

    body_vectors = np.zeros(shape=(batch_image.shape[0], 2, 18))

    for i in range(batch_image.shape[0]):
    	image = batch_image[i]
	    resize_out_ratio = 8.0
	    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

	    if(len(humans)>0):
	        body_vectors[i] = encode_body_vector(humans[0],w, h)
	    else:
	        body_vectors[i] = np.zeros(shape=(2, len(human.body_parts)))
	    # shape = (2, 18)
	    # print(body_vector.shape)
    return body_vectors
