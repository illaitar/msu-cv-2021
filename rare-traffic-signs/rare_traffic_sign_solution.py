# -*- coding: utf-8 -*-
import torch
import torchvision
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

import os
import csv
import json
import tqdm
import pickle
import typing

from PIL import Image

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

CLASSES_CNT = 205

def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)

# -*- coding: utf-8 -*-
import torch
import torchvision
import pytorch_lightning as pl
import albumentations as A
#from albumentations.pytorch import ToTensorV2
 
import os
import csv
import json
import tqdm
import pickle
import typing
 
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier
 
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
import pandas as pd
 
CLASSES_CNT = 205
 
def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)
 
class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        #print('werthj')
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
 
        #-------
        self.samples = []
        for folder in root_folders:
            types = sorted(os.listdir(folder))
            for type_ in types:
                path_to_type = os.path.join(folder, type_)
                for name in sorted(os.listdir(path_to_type)):
                    self.samples.append((os.path.join(path_to_type, name), self.class_to_idx[type_]))
        #print('samples',self.samples)
 
        #-------
        self.classes_to_samples = dict()
        for i in range(len(self.classes)):
            self.classes_to_samples[i] = list()
 
        for i, elem in enumerate(self.samples):
            self.classes_to_samples[elem[1]].append(i)
        #print(self.classes_to_samples)
        #-------
        self.transform = A.Compose([ A.augmentations.transforms.MotionBlur(p=0.25) ,A.augmentations.geometric.rotate.Rotate(20,p=0.5), A.augmentations.Normalize(),
         A.augmentations.Resize(135,135), A.augmentations.crops.transforms.RandomCrop(128,128)])
 
        # self.samples = ... ### YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        # self.classes_to_samples = ... ### YOUR CODE HERE - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        # self.transform = ... ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2
 
    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        path, class_indx = self.samples[index]
        image = imread(path).astype(np.float32)
        image = self.transform(image=image)
        image = torch.from_numpy(image['image'].transpose(2, 0, 1))
        return image, path, class_indx
 
    def __len__(self):
        return len(self.samples)
 
    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        f = open(path_to_classes_json)
        classes_json = json.load(f)
        class_to_idx = dict()
        for i, key in enumerate(classes_json.keys()):
            class_to_idx[key] = i
        # class_to_idx = ... ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
 
        # classes = ... ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        classes = []
        for i, key in enumerate(classes_json.keys()):
            classes.append(key)
        # classes = np.array(classes)
        #print("classes, class_to_idx",classes, class_to_idx)
        return classes, class_to_idx
 
 
class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def __init__(self, root, path_to_classes_json, annotations_file = None):
        super(TestData, self).__init__()
        self.root = root
        #self.samples = ... ### YOUR CODE HERE - список путей до картинок
        self.samples = []
        for name in sorted(os.listdir(root)):
            #self.samples.append(os.path.join(root, name))
            self.samples.append(name)
 
        #self.transform = ... ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        self.transform = A.Compose([A.augmentations.Normalize(), A.augmentations.Resize(128,128)])
 
        classes, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
 
        self.targets = None
        if annotations_file is not None:
            #self.targets = ... ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
            self.targets = dict()
            df = pd.read_csv(annotations_file)
            for path_to_img in self.samples:
                name = path_to_img.split('/')[-1]
                self.targets[path_to_img] = class_to_idx[df[df['filename'] == name].iloc[0, 1]]
 
 
 
    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        name = self.samples[index]
        path = os.path.join(self.root, name)
        image = imread(path).astype(np.float32)
        image = self.transform(image=image)['image']
        image = torch.from_numpy(image.transpose(2, 0, 1))
        target = -1
        if self.targets is not None:
            #print(target)
            target = self.targets[name]
 
        return image, name, target
    def __len__(self):
        return len(self.samples)

class CustomNetwork(pl.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """
    def __init__(self, features_criterion = None, internal_features = 1024):
        super(CustomNetwork, self).__init__()
        
        backbone = torchvision.models.efficientnet_b4(pretrained=False)
        #num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Linear(1792, internal_features)
        self.classifier1 = torch.nn.ReLU()
        self.classifier2 = torch.nn.Linear(internal_features, 205)

        # self.classifier = torch.nn.Linear(num_filters, 205)
        # self.classifier1 = torch.nn.ReLU()
        # self.classifier2 = torch.nn.Linear(internal_features, 205)
        self.loss = features_criterion

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            x = self.feature_extractor(x).flatten(1)
        x = self.classifier(x)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

    def training_step(self, batch, batch_idx):
        """the full training loop"""
        #print(batch)
        x, path, y = batch
        y_pred = self(x)
       
        loss = self.loss(y_pred, y)
        y_pred = torch.argmax(y_pred, axis = 1)
        return {'loss' : loss, 'y_pred' : y_pred.detach(), 'target' : y}

    def training_epoch_end(self, outputs):
        accuracy = []
        for out in outputs:
            accuracy.append(calc_metric(out['y_pred'], out['target'], 'all', None))
        accuracy = np.mean(accuracy)
        print(f"\nTrain accuracy: {accuracy}\n")

    def validation_step(self, batch, batch_idx):
        """the full training loop"""
        #print(batch)
        x, path, y = batch
        y_pred = self(x)
       
        loss = self.loss(y_pred, y)
        y_pred = torch.argmax(y_pred, axis = 1)
        return {'loss' : loss, 'y_pred' : y_pred.detach(), 'target' : y}

    def validation_epoch_end(self, outputs):
        accuracy = []
        for out in outputs:
            accuracy.append(calc_metric(out['y_pred'], out['target'], 'all', None))
        accuracy = np.mean(accuracy)
        print(f"\nVal accuracy: {accuracy}\n")

    def test_step(self, batch, batch_idx):
        x, path, y = batch
        predict = self(x)
        classes = torch.argmax(predict, axis = 1)
        print(classes,y)
        return

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [lr_scheduler]


def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
    model = CustomNetwork(torch.nn.CrossEntropyLoss())
    ds_train = DatasetRTSD(['cropped-train'], 'classes.json')
    trainer = pl.Trainer(
            max_epochs=8,
            gpus=1,
            checkpoint_callback=False,
            logger = False
        )
    print(len(ds_train))
    train_set, val_set = torch.utils.data.random_split(ds_train, [63917, 15979])
    train = DataLoader(train_set, batch_size=64,num_workers=16, shuffle=True)
    val = DataLoader(val_set, num_workers=16,batch_size=64)
    trainer.fit(model, train, val)
    return model


def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    ds = TestData(test_folder, path_to_classes_json)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    out = []
    classes, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
    for elem in dl:
        with torch.no_grad():
            res = np.argmax(model(elem[0]),axis=1)
        for img, path, index, inf in zip(*elem, res):
            d = {}
            d["filename"] = path
            d["class"] = classes[inf]
            out.append(d)
    
    #results = ... ### YOUR CODE HERE - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    results = out
    return results

def test_classifier(output_file, gt_file, classes_file):
    output = read_csv(output_file)
    gt = read_csv(gt_file)
    y_pred = []
    y_true = []
    for k, v in output.items():
        y_pred.append(v)
        y_true.append(gt[k])

    with open(classes_file, "r") as fr:
        classes_info = json.load(fr)
    class_name_to_type = {k: v['type'] for k, v in classes_info.items()}

    total_acc = calc_metric(y_true, y_pred, 'all', class_name_to_type)
    rare_recall = calc_metric(y_true, y_pred, 'rare', class_name_to_type)
    freq_recall = calc_metric(y_true, y_pred, 'freq', class_name_to_type)
    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    :param background_path: путь до папки с изображениями фона
    """
    def __init__(self, background_path):
        ### YOUR CODE HERE
        self.bg_list = os.listdir(background_path)
        self.bg_path = background_path
        self.transforms = A.Compose([A.augmentations.geometric.rotate.Rotate([-16,16],p=1,border_mode=0),
                                     A.augmentations.transforms.ColorJitter(brightness=0.87, contrast=0.4, saturation=0.87, hue=0.07,p=1),
                                    A.augmentations.transforms.MotionBlur(p=0.9),
                                    A.augmentations.transforms.GaussianBlur((1,5),p=0.9)])
        
    def get_sample(self, icon):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        :param icon: Массив с изображением иконки
        """
        r_size = np.random.randint(16,128)
        icon = resize(icon,(r_size,r_size))
        pad_size = np.random.uniform(0,15)
        pad_size = int(r_size*0.01* pad_size)
        icon = np.pad(icon, pad_size)[:,:,pad_size:pad_size+4]
        icon, mask = icon[:,:,0:3], icon[:,:,3:4]
        transformed = self.transforms(image = icon, mask = mask)
        icon = transformed['image']
        mask = transformed['mask']
        mask = np.dstack((mask,mask,mask))
        bg = imread(os.path.join(self.bg_path,np.random.choice(self.bg_list))).astype(np.float32)/255
        tr = A.augmentations.crops.transforms.RandomCrop(int(r_size*1.45),int(r_size*1.45))
        bg = tr(image=bg)['image']
        w,h,c = bg.shape
        pos1 = np.random.randint(0, w - mask.shape[0])
        pos2 = np.random.randint(0, h - mask.shape[1])
        full_mask = np.zeros((w,h,3))
        full_mask[pos1:pos1+mask.shape[0],pos2:pos2+mask.shape[1],:] = mask
        full_icon = np.zeros((w,h,3))
        full_icon[pos1:pos1+mask.shape[0],pos2:pos2+mask.shape[1],:] = icon
        result = full_mask * full_icon + (1-full_mask)*bg
        return result


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    path, out_path, bg_path, num_samples = args[:]
    icon = imread(path).astype(np.float32)/255
    sg = SignGenerator(bg_path)
    lst = os.listdir(out_path)
    for i in range(num_samples):
        img = np.clip(sg.get_sample(icon)*255,0,255).astype('uint8')
        im = Image.fromarray(img)
        fold = path.split('\\')[-1].split('.png')[0]
        new_path = os.path.join(out_path, fold)
        if fold not in lst:
            os.mkdir(new_path)
            lst = os.listdir(out_path)
        im.save(os.path.join(new_path,str(i) + ".png"))
        #imsave(), img)


def generate_all_data(output_folder, icons_path, background_path, samples_per_class = 1100):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и ситетических данных."""
    ### YOUR CODE HERE
    model = CustomNetwork(torch.nn.CrossEntropyLoss())
    ds_train = DatasetRTSD(['newly'], 'classes.json')
    #ds_train2 = DatasetRTSD(['output'], 'classes.json')
    trainer = pl.Trainer(
            max_epochs=4,
            gpus=1,
            checkpoint_callback=False,
            logger = False
        )
    #from torch.utils.data import ConcatDataset
    #ds_train = ConcatDataset([ds_train1, ds_train2])
    print(len(ds_train))
    train_set, val_set = torch.utils.data.random_split(ds_train, [274857, 30539])
    train = DataLoader(train_set, batch_size=64, num_workers=16,shuffle=True)
    val = DataLoader(val_set, num_workers=16, batch_size=64)
    trainer.fit(model, train, val)
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """
    def __init__(self, margin: float) -> None:
        super(FeaturesLoss, self).__init__()
        ### YOUR CODE HERE
        pass


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        ### YOUR CODE HERE
        pass
    def __iter__(self):
        ### YOUR CODE HERE
        pass


def train_better_model():
    """Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки."""
    ### YOUR CODE HERE
    return model


class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors):
        ### YOUR CODE HERE
        pass

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE
        pass

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE
        pass

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        features, model_pred = ... ### YOUR CODE HERE - предсказание нейросетевой модели
        features = features / np.linalg.norm(features, axis=1)[:, None]
        knn_pred = ... ### YOUR CODE HERE - предсказание kNN на features
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    def __init__(self, data_source, examples_per_class) -> None:
        ### YOUR CODE HERE
        pass
    def __iter__(self):
        """Функция, которая будет генерировать список индексов элементов в батче."""
        return ### YOUR CODE HERE


def train_head(nn_weights_path, examples_per_class = 20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE

if __name__ == '__main__':
    model = train_simple_classifier()
    torch.save(model, 'model.ckpt')
    # model = train_synt_classifier()
    # torch.save(model, 'model_synt.ckpt')
    model = torch.load("model.ckpt")
    torch.save(model.state_dict(), "./simple_model.pth")