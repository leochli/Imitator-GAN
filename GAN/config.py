c_dim = 24

image_size = 416
g_conv_dim = 128
d_conv_dim = 128
g_repeat_num = 6
d_repeat_num = 6
lambda_app = 1
lambda_pose = 1
lambda_rec = 1
lambda_gp = 1

# Training configurations.

batch_size = 4
num_iters = 2000
num_iters_decay = 1000
g_lr = 0.01
d_lr = 0.001
n_critic = 1
beta1 = 0.9
beta2 = 0.999  # adam 0.9, 0.999
resume_iters = None

test_iters = None
use_tensorboard = False

# Directories.
pose_data_root = '/home/lichenghui/mpii_human_pose'
deepfashion_data_root = '/home/lichenghui/processed_deep_fashion'
log_dir = '../runs'
sample_dir = '../samples'
model_save_dir = '../saved_model'
result_dir = '../test_result'

# Step size.
log_step = 1
sample_step = 1
model_save_step = 1000
lr_update_step = 500

dumped_model = "model_10_final.pth.tar"
inter_dim = 512
categories = 20