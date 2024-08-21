"""
@DESCRIPTION: Train GIAA model
@AUTHOR: yzc-ippl
"""
import torch
from torch import nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
import copy
from tqdm import tqdm
import warnings
from skimage import transform
from torch.utils.data.dataloader import default_collate

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_gpu = True


class GIAA_model(nn.Module):
    def __init__(self, backbone, out_dim=2048):
        super(GIAA_model, self).__init__()

        #  create backbone
        self.backbone = backbone
        self.backbone.fc = nn.Sequential()

        # create regression
        self.regression = nn.Sequential(nn.Linear(out_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        y = self.regression(x)
        return y


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


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class FlickrDataset_GIAA(Dataset):
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
        # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        rating = self.images_frame.iloc[idx, 1:]
        sample = {'image': image, 'rating': rating}

        if self.transform:
            sample = self.transform(sample)
        return sample


def load_data():
    data_dir = os.path.join(r'./FlickrAES_GIAA/label')
    data_image_dir = os.path.join(r'./FlickrAES_GIAA/image')
    data_image_train_dir = os.path.join(data_dir, 'FLICKR-AES_GIAA_train.csv')
    data_image_test_dir = os.path.join(data_dir, 'FLICKR-AES_GIAA_test.csv')

    transformed_dataset_train = FlickrDataset_GIAA(
        csv_file=data_image_train_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [Rescale(output_size=(256, 256)),
             RandomHorizontalFlip(0.5),
             RandomCrop(
                 output_size=(224, 224)),
             Normalize(),
             ToTensor(),
             ])
    )
    transformed_dataset_valid = FlickrDataset_GIAA(
        csv_file=data_image_test_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [Rescale(output_size=(224, 224)),
             Normalize(),
             ToTensor(),
             ])
    )
    data_train = DataLoader(transformed_dataset_train, batch_size=64,
                            shuffle=True, num_workers=8, collate_fn=my_collate, drop_last=False)
    data_valid = DataLoader(transformed_dataset_valid, batch_size=64,
                            shuffle=True, num_workers=8, collate_fn=my_collate, drop_last=False)

    return data_train, data_valid


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate = 0.5 ** (epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer


def train_GIAA():
    # parameter setting
    num_epoch = 10
    BestSRCC = -10

    # data
    data_train, data_valid = load_data()

    # model
    backbone = models.ResNet50(pretrained=True)
    # backbone = models.resnext101_32x8d(pretrained=True)
    model = GIAA_model(backbone)
    model.cuda()

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5E-2)

    for epoch in range(num_epoch):
        for phase in ['train', 'valid']:
            if phase == 'train':
                print('***********************train***********************')
                model.train()
                optimizer = exp_lr_scheduler(optimizer, epoch)
                loop = tqdm(enumerate(data_train), total=len(data_train), leave=True)
                for batch_idx, data in loop:
                    inputs = data['image']
                    batch_size = inputs.size()[0]
                    labels = data['rating'].view(batch_size, -1)
                    if use_gpu:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    criterion = nn.MSELoss()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    loop.set_description(f'Epoch [{epoch}/{num_epoch}]--train')
                    loop.set_postfix(loss=loss.item())
            if phase == 'valid':
                print('***********************valid***********************')
                model.eval()
                predicts_score = []
                ratings_score = []
                for batch_idx, data in enumerate(data_valid):
                    inputs = data['image']
                    batch_size = inputs.size()[0]
                    labels = data['rating'].view(batch_size, -1)
                    if use_gpu:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    with torch.no_grad():
                        outputs = model(inputs)
                    outputs = outputs.data.cpu().numpy()
                    labels = labels.data.cpu().numpy()
                    predicts_score += outputs.tolist()
                    ratings_score += labels.tolist()
                srcc = spearmanr(predicts_score, ratings_score)[0]
                print('Valid Regression SRCC:%4f' % srcc)
                if srcc > BestSRCC:
                    BestSRCC = srcc
                    best_model = copy.deepcopy(model)
                    torch.save(best_model.cuda(), './model/GIAA.pt')


if __name__ == '__main__':
    train_GIAA()