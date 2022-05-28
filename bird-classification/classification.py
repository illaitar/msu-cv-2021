import os

import numpy as np
from PIL import Image
from tqdm import tqdm


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import models, transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

standart_transform = transforms.Compose([
    transforms.Resize(356),
    transforms.CenterCrop(310),
    transforms.ToTensor(),
    normalize
])


train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=20, shear=0.05),
    transforms.RandomRotation(70),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(356),
    transforms.CenterCrop(310),
    transforms.ToTensor(),
    normalize
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = 0

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res

class Train(Dataset):
    def __init__(self, mode, dataset_path, gt, transformations=standart_transform,
                 class_size=50, train_split=0.8):
        if mode not in ["train", "val"]:
            raise ValueError("Wrong dataset type")

        self.transformations = transformations
        self._items = []

        gt = np.array(list(gt.items()))
        gt_arr = np.zeros(gt.shape, dtype="object")
        for i in range(gt.shape[0]):
            gt_arr[i, :] = [os.path.join(dataset_path, gt[i, 0]), int(gt[i, 1])]
        classes = np.split(gt_arr, class_size)

        indexes = slice(None, int(class_size * train_split)) if mode == "train" else \
            slice(int(class_size * train_split), None)

        for c in classes:
            self._items += list(c[indexes])

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, label = self._items[index]

        img = Image.open(img_path).convert('RGB')
        if self.transformations is not None:
            return self.transformations(img), label

        return img, label


class Test(Dataset):
    def __init__(self, dataset_path, transformations=standart_transform):
        self.transformations = transformations
        self._items = []

        filenames = sorted(list(os.walk(dataset_path))[0][2])
        for filename in filenames:
            self._items.append(os.path.join(dataset_path, filename))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path = self._items[index]
        img = Image.open(img_path).convert('RGB')
        return self.transformations(img), img_path


class ModuleData(pl.LightningDataModule):
    def __init__(self, train_path, train_gt,
                 batch_size=32, train_transforms=standart_transform, val_transforms=standart_transform):
        super().__init__()

        self.batch_size = batch_size
        self.train_set = Train("train", train_path, train_gt, transformations=train_transforms)
        self.val_set = Train("val", train_path, train_gt, transformations=val_transforms)


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

class Densenett(pl.LightningModule):
    def __init__(self, num_classes, transfer=True, freeze='most'):
        super().__init__()
        self.mobilenet_model = models.efficientnet_b4(pretrained=transfer)
        #print(self.mobilenet_model)
        self.mobilenet_model.classifier = nn.Sequential(
            nn.Linear(1792 , 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        for child in list(self.mobilenet_model.children()):
            for param in child.parameters():
                param.requires_grad = True

        if freeze == 'last':
            for child in list(self.mobilenet_model.children())[0][:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze == 'most':
            for child in list(self.mobilenet_model.children())[0][:-3]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze != 'full':
            raise NotImplementedError('Wrong freezing parameter')

    def _forward_impl(self, x):
        x = self.mobilenet_model.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = F.softmax(self.mobilenet_model.classifier(x), dim=1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

class MyModel3(pl.LightningModule):
    def __init__(self, lr_rate=1e-3, freeze="most", fast_train=False):
        super(MyModel3, self).__init__()
        self.model = Densenett(50, transfer=not fast_train, freeze=freeze)
        self.lr_rate = lr_rate
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def predict_dict(self, dataloader):
        preds = dict()
        for i, data in enumerate(tqdm(dataloader)):
            images, img_paths = data
            with torch.no_grad():
                logits = self.forward(images.to(device))
                pred = logits.argmax(dim=1).cpu().numpy()
                for k in range(len(pred)):
                    preds[img_paths[k].split('/')[-1]] = int(pred[k])
        return preds

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        logs = {
            'train_loss': loss.detach(),
            'train_acc': acc.detach()
        }
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        self.log('val_loss', loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc, on_step=True, on_epoch=False)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        self.log('test_loss', loss, on_step=True, on_epoch=False)
        self.log('test_acc', acc, on_step=True, on_epoch=False)
        return {'test_loss': loss, 'test_acc': acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['log']['train_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['log']['train_acc'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, on_epoch=True, on_step=False)
        self.log('train_acc', avg_acc, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_epoch=True, on_step=False)
        self.log('val_acc', avg_acc, on_epoch=True, on_step=False)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, on_epoch=True, on_step=False)
        self.log('test_acc', avg_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_rate)
        lr_scheduler = StepLR(optimizer, step_size=12, gamma=0.1)
        return [optimizer], [lr_scheduler]

def train_classifier(train_gt, train_img_dir, fast_train=False):
    """ Callbacks and Trainer """

    checkpoint_callback = ModelCheckpoint(dirpath='./',
                                          filename='{epoch}-{val_acc:.3f}',
                                          monitor='val_acc', mode='max', save_top_k=1, verbose=True)

    max_epochs = 1 if fast_train else 25
    callbacks = [] if fast_train else [checkpoint_callback]
    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        callbacks=callbacks,
        max_epochs=max_epochs,
        logger=not fast_train,
        checkpoint_callback=not fast_train
    )

    data_module = ModuleData(train_img_dir, train_gt, train_transforms=train_transform)
    model = MyModel3(fast_train=fast_train).to(device)
    trainer.fit(model, data_module)

    return model


def classify(model_path, test_img_dir):
    model = MyModel3(fast_train = True)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval().to(device)
    ds_test = Test(dataset_path=test_img_dir)
    dl_test = DataLoader(ds_test, batch_size=32, shuffle=False)
    ast = model.predict_dict(dl_test)
    return ast