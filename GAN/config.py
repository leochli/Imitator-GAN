# c_dim = 24
c_dim = 20

image_size = 416
g_conv_dim = 64
d_conv_dim = 64
g_repeat_num = 4
d_repeat_num = 4
lambda_app = 5
lambda_pose = 0.00001
lambda_rec = 1
lambda_gp = 1

# Training configurations.

batch_size = 4
num_iters = 200000
num_iters_decay = 150000
g_lr = 0.0001
d_lr = 0.0001
n_critic = 1
beta1 = 0.9
beta2 = 0.999  # adam 0.9, 0.999
resume_iters = 18000

test_iters = None
use_tensorboard = False

# Directories.
pose_data_root = '/home/lichenghui/mpii_human_pose'
deepfashion_data_root = '/home/lichenghui/processed_deep_fashion_full'
log_dir = '../runs'
sample_dir = '../samples'
model_save_dir = '../saved_model'
result_dir = '../test_result'

# Step size.
log_step = 20
sample_step = 50
model_save_step = 10000
lr_update_step = 1000

dumped_model = "model_10_final.pth.tar"
inter_dim = 512
categories = 20