import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import cv2
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import albumentations as A
import albumentations_experimental

def to_img(ten_img):
    return np.array(ten_img).transpose((2, 1, 0)).transpose(1,0,2)
def to_ten(st_img):
    return torch.from_numpy(np.array(st_img.copy()).transpose((1, 0, 2)).transpose(2,1,0))
def to_standart_keys(points):
    part1 = points[::2]
    part2 = points[1::2]
    return [(part1[i], part2[i]) for i in range(14)]
def to_label(keys):
    res = np.zeros(28, dtype='float32')
    res[::2] = np.array(keys)[:,0]
    res[1::2] = np.array(keys)[:,1]
    return  torch.from_numpy(res)

class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mean = [0.5346, 0.4285, 0.3732]
        self.std = [0.2352, 0.2153, 0.2055]
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        ind = str(idx).zfill(5) + '.jpg'
        img_path = os.path.join(self.img_dir, ind)
        image = read_image(img_path)
        label = self.img_labels[ind].copy()
        label = np.abs(label)
        scale_x,scale_y  = image.shape[1:]
        scale_x /= 128
        scale_y /= 128
        label[::2] /= scale_y
        label[1::2] /= scale_x
        label = np.array(label,dtype="float32")
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = fn.resize(image, (128,128))
        image = image / 255
        img = to_img(image)
        pts = to_standart_keys(label)
        symm_keypoints = [(0, 3), (1, 2), (4, 9), (5, 8), (6, 7), (11, 13), (10, 10), (12, 12)]
        keypoint_params=A.KeypointParams(format='xy', angle_in_degrees=False)
        tr = albumentations_experimental.augmentations.transforms.HorizontalFlipSymmetricKeypoints(symm_keypoints,p=0.5)
        ttransform = A.Compose([
            tr,
            A.SafeRotate(limit=30,p=0.3)
        ], keypoint_params=keypoint_params)
        zzz = ttransform(image= img,keypoints=pts)
        l1 = zzz['keypoints']
        img1 = zzz['image']
        if len(l1) == 14:
            label = to_label(l1)
            t_img = img1
        else:
            label = label
            t_img = img
        aug =  A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=0.7),
                          A.Blur(2),
                         A.RGBShift (r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5) 
                         ])
        image = aug(image = t_img)
        image = to_ten(t_img)
        
        for k in range(3):
            image[k,:,:] -= self.mean[k]
            image[k,:,:] /= self.std[k]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, torch.Tensor(label)
    def __len__(self):
        return len(self.img_labels)



class TestSet(Dataset):
    def __init__(self, img_dir):
        self.files = sorted(os.listdir(img_dir))
        res = {}
        self.mean = [0.5346, 0.4285, 0.3732]
        self.std = [0.2352, 0.2153, 0.2055]
        self.img_dir = img_dir
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = read_image(img_path)
        imsize = image.shape[1:]
        scale_x,scale_y = image.shape[1:]
        scale_x /= 128
        scale_y /= 128
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = fn.resize(image, (128,128))
        image = image / 255
        for k in range(3):
            image[k,:,:] -= self.mean[k]
            image[k,:,:] /= self.std[k]
        return image, self.files[idx]

class FacialModel(pl.LightningModule):
    def __init__(self, learning_rate=5e-4):
        super().__init__()
        self.layer1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=2),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(25600, 64)
#         self.drop1 = torch.nn.Dropout(0.1)
#         self.fc2 = nn.Linear(128, 64)
        #self.drop2 = torch.nn.Dropout(0.05)
        self.fc3 = nn.Linear(64, 28)
        self.train_accuracy = torch.nn.MSELoss()
        self.val_accuracy = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        out = self.layer1_1(x)
        #print(out.shape)
        out = self.layer1_2(out)
        out = self.layer1_3(out)
        out = self.layer1_4(out)
        out = self.layer1_5(out)
        
        out2 = self.layer2_1(x)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)
        out2 = self.layer2_4(out2)
        out2 = self.layer2_5(out2)
        
       
        out = out.reshape(out.size(0), -1)
        out2 = out.reshape(out2.size(0), -1)
       
        mid_res = torch.cat((out,out2),dim=1)
        #print(out.shape)
        #print(mid_res.shape)
        out = self.fc1(mid_res)
        
        #out = self.drop2(out)
        out = self.fc3(out)
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        ls = torch.nn.MSELoss()
        loss = ls(y_pred, y)
        train_acc_batch = self.train_accuracy(y_pred, y)
        return {'loss' : loss, 'y_pred' : y_pred.detach(), 'target' : y}
    
    def training_epoch_end(self, outputs):
        accuracy = []
        for out in outputs:
            accuracy.append(self.train_accuracy(out['y_pred'], out['target']))
        accuracy = torch.mean(torch.stack(accuracy))
        print(f"Train Loss: {accuracy}")
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        ls = torch.nn.MSELoss()
        loss = ls(y_pred, y)
        val_acc_batch = self.val_accuracy(y_pred, y)
        return {'loss' : loss, 'y_pred' : y_pred.detach(), 'target' : y}
    
    def validation_epoch_end(self, outputs):
        accuracy = []
        for out in outputs:
            accuracy.append(self.val_accuracy(out['y_pred'], out['target']))
        accuracy = torch.mean(torch.stack(accuracy))
        print(f"Validation Loss: {accuracy}")


def train_detector(train_gt, train_img_dir, fast_train):
    training_data = MyDataset(train_gt, train_img_dir)
    train_set, val_set = torch.utils.data.random_split(training_data, [5800, 200])
    train = DataLoader(train_set, batch_size=8, shuffle=True)
    val = DataLoader(val_set, batch_size=8)
    model = FacialModel()
    trainer = pl.Trainer(max_epochs=100)
    if fast_train:
        trainer = pl.Trainer(max_epochs=1,checkpoint_callback=False,logger=False)
        trainer.fit(model, val)
    else:
        trainer.fit(model, train, val)
    # torch.save(model, 'facepoints_model.ckpt')
    print("##############")
    print("all right! :)")
    print("##############")

def detect(model_filename, test_img_dir):
    model = FacialModel().load_from_checkpoint(checkpoint_path=model_filename)
    #model.eval()
    test_data = TestSet(test_img_dir)
    tests = DataLoader(test_data, batch_size = 8)
    res = {}
    for test, file in tests:
        print(file[0])
        shapes = []
        for z in file:
            img_path = os.path.join(test_img_dir, z)
            image = read_image(img_path)
            shapes.append(image.shape[1:])
        with torch.no_grad():
            pred = model(test)
        for k in range(len(file)):
            scale_x,scale_y  = shapes[k]
            scale_x /= 128
            scale_y /= 128
            pred[k][::2] *= scale_y
            pred[k][1::2] *= scale_x
            res[file[k]] = pred[k].cpu().detach().numpy()
    return res
    