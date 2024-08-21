# -*- coding: utf-8 -*-
"""
@DESCRIPTION: Test PIAA model
@AUTHOR: yzc-ippl
"""
from __future__ import print_function, division
import os
import torch
import warnings
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import random
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from PIL import Image
import time
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from GIAA.train_GIAA_model import GIAA_model
from MTCL.train_Contrast_model import Contrast_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True


class FlickrAESDataset_PIAA(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
        img_name = str(os.path.join(self.root_dir, str(self.images_frame.iloc[idx, 0])))
        im = Image.open(img_name).convert('RGB')
        if im.mode == 'P':
            im = im.convert('RGB')
        image = np.asarray(im)
        rating = self.images_frame.iloc[idx, 1:]
        sample = {'image': image, 'rating': rating}

        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image / 1.0  # / 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}



class PIAA_model(nn.Module):
    def __init__(self, Contrast_model, GIAA_model):
        super(PIAA_model, self).__init__()

        # self.Contrast_model_encoder = Contrast_model.module.encoder
        self.Contrast_model_encoder = Contrast_model.encoder
        self.GIAA_model = GIAA_model.backbone
        self.tahn = nn.Tanh()
        for param in self.GIAA_model.parameters():
            param.requires_grad = False
        self.regression = GIAA_model.regression
        # for param in self.Contrast_model_encoder.parameters():
        #     param.requires_grad = False
        for param in self.regression.parameters():
            param.requires_grad = True

    def forward(self, x):
        GIAA_feature = self.GIAA_model(x)
        Contrast_feature = self.Contrast_model_encoder(x)
        PIAA_feature = self.tahn(Contrast_feature) + GIAA_feature
        PIAA_score = self.regression(PIAA_feature)
        return PIAA_score


def computeSpearman(dataloader_valid, model, net_2, epoch):
    ratings = []
    predictions = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_valid):
            inputs = data['image']
            batch_size = inputs.size()[0]
            labels = data['rating'].view(batch_size, -1)
            labels = labels / 5.0
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                except:
                    print(inputs, labels)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            if epoch == 0:
                # outputs_a, _ = net_2(inputs)
                outputs_a = net_2(inputs)
            else:
                # outputs_a, _ = model(inputs)
                outputs_a = model(inputs)
            labels = labels.data.cpu().numpy()
            outputs_a = outputs_a.data.cpu().numpy()
            ratings.append(labels)
            predictions.append(outputs_a)

    ratings_i = np.vstack(ratings)
    predictions_i = np.vstack(predictions)
    a = ratings_i[:, 0]
    b = predictions_i[:, 0]
    sp = spearmanr(a, b)
    return sp


def train_model(model, criterion, optimizer, dataloader_train, dataloader_valid, num_epochs=100):
    since = time.time()
    best_spearman = 0
    criterion.cuda()
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                for batch_idx, data in enumerate(dataloader_train):
                    inputs = data['image']
                    batch_size = inputs.size()[0]
                    labels = data['rating'].view(batch_size, -1)
                    labels = labels / 5.0
                    if use_gpu:
                        try:
                            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                        except:
                            print(inputs, labels)
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            if phase == 'val':
                model.eval()
                ratings = []
                predictions = []
                for batch_idx, data in enumerate(dataloader_valid):
                    inputs = data['image']
                    batch_size = inputs.size()[0]
                    labels = data['rating'].view(batch_size, -1)
                    labels = labels / 5.0
                    if use_gpu:
                        try:
                            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                        except:
                            print(inputs, labels)
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    with torch.no_grad():
                        outputs = model(inputs)
                    labels = labels.data.cpu().numpy()
                    outputs_a = outputs.data.cpu().numpy()
                    ratings.append(labels)
                    predictions.append(outputs_a)

                ratings_i = np.vstack(ratings)
                predictions_i = np.vstack(predictions)
                a = ratings_i[:, 0]
                b = predictions_i[:, 0]
                spearman = spearmanr(a, b)[0]

                print(spearman)
                if spearman > best_spearman:
                    best_spearman = spearman
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Val Spearman: {:4f}'.format(best_spearman))
    return best_spearman


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def Flicker_test():
    data_dir = os.path.join('./FlickerAes_PIAA/label')

    workers_test = pd.read_csv(os.path.join(data_dir, 'test_worker.csv'), sep=' ')
    worker_test_orignal = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_each_worker.csv'), sep=',')

    workers_fold = "Workers_test/"
    if not os.path.exists(workers_fold):
        os.makedirs(workers_fold)

    epochs = 10
    spp = []
    # meta_num = 100
    meta_num = 10
    for worker_idx in range(37):

        worker = workers_test['worker'].unique()[worker_idx]
        print("----worker number: %2d---- %s" % (worker_idx, worker))
        num_images = worker_test_orignal[worker_test_orignal['worker'].isin([worker])].shape[0]
        percent = meta_num / num_images
        # print('train image: %d, test image: %d' % (100, num_images - 100))
        print('train image: %d, test image: %d' % (10, num_images - 10))
        images = worker_test_orignal[worker_test_orignal['worker'].isin([worker])][[' imagePair', ' score']]

        srocc_list = []
        for i in range(0, 10):
            train_dataframe, valid_dataframe = train_test_split(images, train_size=percent)
            train_path = workers_fold + "train_scores_" + worker + ".csv"
            test_path = workers_fold + "test_scores_" + worker + ".csv"
            train_dataframe.to_csv(train_path, sep=',', index=False)
            valid_dataframe.to_csv(test_path, sep=',', index=False)

            output_size = (224, 224)
            transformed_dataset_train = FlickrAESDataset_PIAA(csv_file=train_path,
                                                            root_dir='./FlickerAes_PIAA/image',
                                                            transform=transforms.Compose(
                                                                [Rescale(output_size=(256, 256)),
                                                                 RandomHorizontalFlip(0.5),
                                                                 RandomCrop(
                                                                     output_size=output_size),
                                                                 Normalize(),
                                                                 ToTensor(),
                                                                 ]))
            transformed_dataset_valid = FlickrAESDataset_PIAA(csv_file=test_path,
                                                            root_dir='./FlickerAes_PIAA/image',
                                                            transform=transforms.Compose(
                                                                [Rescale(output_size=(224, 224)),
                                                                 Normalize(),
                                                                 ToTensor(),
                                                                 ]))
            bsize = meta_num
            dataloader_train = DataLoader(transformed_dataset_train, batch_size=bsize,
                                          shuffle=False, num_workers=0, collate_fn=my_collate)
            dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=250,
                                          shuffle=False, num_workers=0, collate_fn=my_collate)


            giaa_model = torch.load('./model/GIAA.pt')
            contrast_model = torch.load('./model/Contrast.pt')

            piaa_model = PIAA_model(contrast_model, giaa_model)

            model_ft = piaa_model
            model_ft.cuda()

            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model_ft.parameters(), lr=0.0001, weight_decay=5E-2)
            spearman = train_model(model_ft, criterion, optimizer, dataloader_train,
                                   dataloader_valid, num_epochs=epochs)
            srocc_list.append(spearman)
        srocc = np.mean(srocc_list)
        print("-------average srocc is: %4f-------" % srocc)
        spp.append(srocc)
    print(spp)
    print(np.mean(spp))


if __name__ == '__main__':
    Flicker_test()
