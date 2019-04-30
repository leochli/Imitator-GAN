from gan import Generator
from gan import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from feature_net import f_model
from utils import *
from dataloader import *
from data_process import *
import config

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.c_dim = config.c_dim

        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_app = config.lambda_app
        self.lambda_pose = config.lambda_pose
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        # self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        # self.selected_attrs = config.selected_attrs

        # loss functions
        self.feature_loss = torch.nn.CosineEmbeddingLoss()
        self.pose_loss = torch.nn.PairwiseDistance()

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

        """Build the feature extractor"""
        self.feature_model = f_model(model_path=DUMPED_MODEL, freeze_param=True).cuda()#.cuda()
        self.feature_model.eval()


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


    def normalize(self, x):
        pass

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)
        pass

    def feat_extract(self, resized_data):
        # input: N * 3 * 224 * 224
        resized_data = resized_data.to(self.device)#.cuda()
        # output: N * num_classes, N * inter_dim, N * C' * 7 * 7
        out, inter_out, x = self.feature_model(resized_data)
        # return batch feature vector
        return out.clamp_(0, 1)

    def pose_extract(self, batch_img, bbx=None):
        # input [N, 224, 224, 3] in RGB
        # output [N, 2, 18]
        # permute = [2, 1, 0]
        # # crop_batch = np.zeros(shape=())
        # for i in range(batch_img.shape[0]):
        #     img = crop(batch_img[i], bbx[i])
        #     # RGB TO BGR
        #     img = img[:, permute, : , :]
        #     crop_batch[i, :, :] = img
        # pose_vectors = get_batch_body_vector(crop_batch.numpy())

        batch_img = np.transpose(batch_img, (0, 2, 3, 1))
        # print(batch_img.shape)
        pose_vectors = get_batch_body_vector(batch_img)
        pose_vectors = torch.from_numpy(pose_vectors)
        return pose_vectors

    def appreance_cos_similarity(self, feat_fake, feat_real):
        label = torch.ones(feat_real.shape[0]).cuda()
        cos_simi = self.feature_loss(feat_real, feat_fake, label)
        return cos_simi

    def compute_pose_loss(self, pose_fake, pose_real, mask):
        mask = mask.view(-1, 1, 18).double()
        pose_fake = torch.mul(pose_fake, mask)
        pose_real = torch.mul(pose_real.double(), mask)
        distance = self.pose_loss(pose_real, pose_fake)
        return distance.float().sum()



    def train(self):
        """Train StarGAN within a single dataset."""


        # Fetch fixed inputs for debugging.
        data_iter = iter(self.data_loader)
        a_fixed, b_fixed, bbox_fixed, b_fixed_pose_feat, mask_fixed = next(data_iter)
        a_fixed = a_fixed.to(self.device)
        b_fixed = b_fixed.to(self.device)
        bbox_fixed = bbox_fixed.to(self.device)
        # c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for step in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                a_real, b_real, bbox, b_pose_feat, mask = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                a_real, b_real, bbox, b_pose_feat, mask = next(data_iter)


            a_real = a_real.to(self.device)  # Input images.
            b_real = b_real.to(self.device)
            bbox = bbox.to(self.device)
            b_pose_feat = b_pose_feat.to(self.device)
            mask = mask.to(self.device)

            # extract appearance feature
            a_app_feat = self.feat_extract(a_real)
            a_app_feat = a_app_feat.to(self.device)

            # # extract pose feature
            # b_pose_feat = self.pose_extract(b_real)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src = self.D(b_real)
            d_loss_real = - torch.mean(out_src)
            # d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            # con_feat = torch.cat([a_app_feat, bbox/416.0], dim=1)
            con_feat = a_app_feat

            x_fake = self.G(b_real, con_feat)
            out_src = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)
            # fake_app_feat = self.feat_extract(x_fake)
            # fake_pose_feat = self.pose_extract(x_fake, bbox)
            # d_loss_app = self.appreance_cos_similarity(fake_app_feat, a_app_feat)
            # d_loss_pose = - self.pose_loss(fake_pose_feat, b_pose_feat)


            # Compute loss for gradient penalty.
            alpha = torch.rand(b_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * b_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            # d_loss = d_loss_real + d_loss_fake + self.lambda_app * d_loss_cls + self.lambda_gp * d_loss_gp
            # d_loss = d_loss_fake + d_loss_real + self.lambda_app * d_loss_app + self.lambda_pose * d_loss_pose
            # d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
            d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            # loss['D/loss_app'] = d_loss_app.item()
            # loss['D/loss_pose'] = d_loss_pose.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (step + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(b_real, con_feat)
                # print(x_fake[0,:,200:205,200:205])
                out_src = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)

                crop_batch = torch.zeros((x_fake.shape[0], 3, 224, 224))
                b = bbox.detach().cpu().numpy().astype(int)
                for i in range(x_fake.shape[0]):
                    # img = crop(x_fake[i], bbox[i])
                    x1, x2, y1, y2 = b[i,0], b[i,0]+b[i,2], b[i,1], b[i,1]+b[i,3]
                    x1 = min(max(x1, 0), 416)
                    x2 = min(max(x2, 0), 416)
                    y1 = min(max(y1, 0), 416)
                    y2 = min(max(y2, 0), 416)

                    img = x_fake[i, :, x1:x2, y1:y2].cpu().data.numpy()
                    img = img.transpose((1,2,0))
                    resized_img = np.zeros(shape=(224, 224, 3))
                    resized_img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
                    crop_batch[i, :, :, :] = torch.from_numpy(resized_img.transpose((2,0,1)))


                fake_app_feat = self.feat_extract(crop_batch)
                fake_pose_feat = self.pose_extract(crop_batch.numpy())

                # #**** debug ****#
                # fake_images = (x_fake.cpu().data).numpy()
                # permute = [2, 1, 0]
                # fake_images = fake_images[:, permute, :, :].transpose((0,2,3,1))
                # resized_data = np.zeros(shape=(fake_images.shape[0], 224, 224, 3))
                # for j in range(fake_images.shape[0]):
                #     resized_data[j,:,:,:] = cv2.resize(fake_images[j,:,:,:], (224, 224), interpolation = cv2.INTER_AREA)
                # resized_data = np.transpose(resized_data, (0, 3, 1, 2))
                # resized_tensor = torch.from_numpy(resized_data)
                # resized_tensor = resized_tensor.to(self.device, dtype=torch.float)

                # fake_app_feat = self.feat_extract(resized_tensor)
                # fake_pose_feat = self.pose_extract(resized_data, bbox)

                fake_app_feat = fake_app_feat.to(self.device)
                fake_pose_feat = fake_pose_feat.to(self.device)
                #**** debug ****#

                # g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)
                g_loss_app = - self.appreance_cos_similarity(fake_app_feat, a_app_feat)  # -similarity
                # print(fake_pose_feat.size(), b_pose_feat.size(), mask.size())
                g_loss_pose = self.compute_pose_loss(fake_pose_feat, b_pose_feat, mask)  # joints distance


                # Backward and optimize.
                # g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_app * g_loss_cls
                # g_loss = g_loss_fake + self.lambda_app * g_loss_app + self.lambda_pose * g_loss_pose
                g_loss = g_loss_fake + self.lambda_app * g_loss_app + self.lambda_pose * g_loss_pose
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                # loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_app'] = g_loss_app.item() * self.lambda_app
                loss['G/loss_pose'] = g_loss_pose.item() * self.lambda_pose


            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (step + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, step + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (step + 1) % self.sample_step == 0:

                with torch.no_grad():
                    # a fix: [N, 3, 224, 224]
                    # a_real, b_real, bbox, b_pose_feat, mask
                    a_resized = torch.zeros(size=(a_real.shape[0], 3 ,416, 416))
                    for i in range(a_real.shape[0]):
                        img = a_real[i].cpu().data.numpy()
                        img = img.transpose((1,2,0))
                        resized_img = np.zeros(shape=(416, 416, 3))
                        resized_img = cv2.resize(img, (416, 416), interpolation = cv2.INTER_AREA)
                        a_resized[i, :, :, :] = torch.from_numpy(resized_img.transpose((2,0,1)))

                    a_resized = a_resized.to(self.device)

                    picture_list = [a_resized, b_real]
                    a_visual_feat = self.feat_extract(a_real)
                    # a feature: [N, 20]; bbox: [N,4]
                    # con_visual_feat = torch.cat([a_visual_feat, bbox/416.0], dim=1) # [N, 24]
                    con_visual_feat = a_visual_feat
                    # print(b_real, con_visual_feat)
                    x_fake = self.G(b_real, con_visual_feat) # [N, 3, 416, 416]
                    # print(a_fixed.size(), b_fixed.size(), x_fake.size())
                    picture_list.append(x_fake)
                    picture_concat = torch.cat(picture_list, dim=0)
                    # print(picture_concat.size())
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(step + 1))
                    save_image(self.denorm(picture_concat.data.cpu()), sample_path, nrow=4, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (step + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(step + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(step + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (step + 1) % self.lr_update_step == 0 and (step + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.

        data_loader = self.data_loader


        with torch.no_grad():
            for i, (a_real, b_real) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                a_real = a_real.to(self.device)
                b_real = b_real.to(self.device)

                # Translate images.
                a_fake_list = [a_real, b_real]
                a_fixed_feat = self.feat_extract(a_real)
                a_fake_list.append(self.G(b_real, a_fixed_feat))

                # Save the translated images.
                x_concat = torch.cat(a_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))


if __name__ == "__main__":
    pose_data_root = config.pose_data_root
    fashion_data_root = config.deepfashion_data_root
    train_loader = get_dataloader(fashion_data_root, pose_data_root, config.batch_size)
    data_iter = iter(train_loader)
    solve = Solver(train_loader, config)
    solve.train()