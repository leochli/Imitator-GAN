import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import random
import numpy as np


class trainDataset(Dataset):
    def __init__(self, fashion_dir, pose_dir, fashion_transforms, pose_transforms):
        self.fashion_dir = fashion_dir
        self.pose_dir = pose_dir
        fashion_path = os.path.join(fashion_dir, 'annotation', 'image_anno.txt')
        pose_path = os.path.join(pose_dir, 'annotation', 'anno_bbox_pose_full.json')

        self.fashion_transforms = fashion_transforms
        self.pose_transforms = pose_transforms
        with open(fashion_path, 'r') as list_file:
            self.fashion_list = list_file.readlines()

        self.pose_data = json.load(open(pose_path, 'r'))['annotations']
        if len(self.fashion_list) > len(self.pose_data):
            self.fashion_list = random.sample(self.fashion_list, len(self.pose_data))
        else:
            self.pose_data = random.sample(self.pose_data, len(self.fashion_list))

    def __len__(self):
        return len(self.fashion_list)

    def __getitem__(self, index):

        fashion_img_file = os.path.join(self.fashion_dir, 'images', self.fashion_list[index].strip().split('/')[-1])
        fashion_image = Image.open(fashion_img_file).convert('RGB')
        fashion_image = self.fashion_transforms(fashion_image)

        pose_img_path = self.pose_data[index]['image_path'].split('/')[-1]
        pose_img_file = os.path.join(self.pose_dir, 'images', pose_img_path)
        pose_image = Image.open(pose_img_file).convert('RGB')
        pose_image = self.pose_transforms(pose_image)

        mask = self.pose_data[index]['mask']
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        pose_vec = self.pose_data[index]['pose_vector']
        pose_vec = torch.tensor(np.array(pose_vec), dtype=torch.long)
        pose_vec = pose_vec.view(2, 18)
        bbox = self.pose_data[index]['bbox']
        bbox = torch.tensor(np.array(bbox), dtype=torch.float)

        return fashion_image, pose_image, bbox, pose_vec, mask


def get_dataloader(fashion_dir, pose_dir, batch_size):
    fashion_transform = transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor()])
    pose_transform = transforms.Compose([transforms.Resize((416, 416)),
                        transforms.ToTensor()])
    dataset = trainDataset(fashion_dir, pose_dir,fashion_transform, pose_transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader

if __name__ == "__main__":
    train_loader = get_dataloader('../processed_deep_fashion', '../human_pose', 10)
    data_iter = iter(train_loader)
    a_real, b_real, bbox, b_pose_feat, mask = next(data_iter)
    print(a_real.shape)
