import os
import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.models
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class SimDataset(Dataset):
    def __init__(self, train_data_path, transform=None):
        super().__init__()
        self.image_mask_pathes = [
            (os.path.join(train_data_path, "images", cls_name, image_path), 
             os.path.join(train_data_path, "gt", cls_name, mask_path))
            for cls_name in os.listdir(os.path.join(train_data_path, "images"))
            for image_path, mask_path in zip(
                sorted(os.listdir(os.path.join(train_data_path, "images", cls_name))),
                sorted(os.listdir(os.path.join(train_data_path, "gt", cls_name)))
            )
        ]
        self.transform = transform
        if transform is None:
            raise ValueError("No transform provided")

    def __len__(self):
        return len(self.image_mask_pathes)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pathes[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path)).astype(float) / 255
        image_mask = self.transform(image=image, mask=mask)
        mask = image_mask["mask"].unsqueeze(0)
        image = image_mask["image"]
        if mask.shape != (1, 224, 224):
            mask = mask[..., 0]
        return [image, mask]


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class, pretrained):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=pretrained)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +
                                                 target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


class MyModel(pl.LightningModule):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        print("Pretrained:", pretrained)
        self.model = ResNetUNet(num_classes, pretrained)
        for l in self.model.base_layers:
            for param in l.parameters():
                param.requires_grad = False

        self.bce_weight = 0.9

    def forward(self, x):
        """ Use for inference only (separate from training_step). """
        x = self.model(x)
        return x

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(f"| Train_loss: {avg_loss:.3f}")
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['logs']['dice'] for x in outputs]).mean()
        avg_bce = torch.stack([x['logs']['bce'] for x in outputs]).mean()
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.3f}, Val_dice: {avg_dice:.3f}, Val_bce: {avg_bce:.3f}", end=" ")
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)

    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch
        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  mode='min', 
                                                                  factor=0.15, 
                                                                  patience=1, 
                                                                  verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        } 
        return [optimizer], [lr_dict]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)
        return {'val_loss': loss, 'logs': {'dice': dice, 'bce': bce}}


def get_model(pretrained=False):
    return MyModel(pretrained=pretrained)


def train_segmentation_model(train_data_path, pretrained=False):
    dataset = SimDataset(train_data_path=train_data_path, transform=A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()]))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    model = get_model(pretrained)

    dl_train = DataLoader(train_set, batch_size=8, shuffle=True)
    dl_val = DataLoader(val_set, batch_size=8, shuffle=False)

    MyModelCheckpoint = ModelCheckpoint(dirpath='runs/segmentation',
                                        filename='{epoch}-{val_loss:.3f}',
                                        monitor='val_loss', 
                                        mode='min', 
                                        save_top_k=1)

    MyEarlyStopping = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=3,
                                    verbose=True)

    trainer = pl.Trainer(
        max_epochs=20,
        gpus=0,
        callbacks=[MyEarlyStopping, MyModelCheckpoint]
    )
    trainer.fit(model, dl_train, dl_val)
    return model


def predict(model, filename, device=torch.device("cpu")):
    image = np.array(Image.open(filename).convert("RGB"))
    h, w = image.shape[:2]
    transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    image = transform(image=image)["image"].unsqueeze(0)
    model.eval()
    if device is not None:
        model = model.to(device)
        image = image.to(device)
    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy().squeeze()
    return cv2.resize(pred, (w, h))


if __name__ == '__main__':
    model = torch.load('segmentation_model.ckpt')
    normalized_state_dict = {}
    for name, weight in model['state_dict'].items():
        mask = (weight.abs() > 1e-32).float()
        normalized_state_dict[name] = weight * mask
    torch.save(normalized_state_dict, "../segmentation_model.pth")
    # train_segmentation_model('tests/00_test_val_input/train')
