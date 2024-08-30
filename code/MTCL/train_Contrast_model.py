# -*- coding: utf-8 -*-
"""
@DESCRIPTION: Train Contrast model
@AUTHOR: yzc-ippl
"""
import sys
import random
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
import warnings
from torch.autograd import Variable
import torch
from torch import nn, optim
from tqdm import tqdm
import copy
from torch.utils.data.dataloader import default_collate
from code.GIAA.train_GIAA_model import GIAA_model
from torchvision import models


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_gpu = True


class FlickrDataset_MTCL(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_information = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_information)

    def __getitem__(self, idx):
        img_path = str(os.path.join(self.root_dir, (str(self.image_information.iloc[idx, 0]))))
        image = Image.open(img_path).convert('RGB')
        label = self.image_information.iloc[idx, 1:]
        if self.transform:
            image = self.transform(image)
        sample = {'name': img_path,
                  'image': image,
                  'label': torch.from_numpy(np.float64([label])).double()}
        return sample


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class Contrast_Database:

    def __init__(self, data_dir='./FlickrAES_TrainUser', num_batch=100, save_dir='./Contrast_Database'):

        self.data_dir = data_dir
        self.num_batch = num_batch
        self.save_dir = save_dir

    def peer_random_select(self, Select_user_ID='A14W0IW2KGR80K', Select_PIAA_score=5):
        csv_name = str(Select_user_ID) + '.csv'
        worker_data = pd.read_csv(os.path.join(self.data_dir, csv_name), sep=',')
        all_data = []
        all_score = []
        for index, row in worker_data.iterrows():
            image_score = int(row[1])
            all_data.append(np.array(row))
            all_score.append(image_score)
        all_score = np.array(all_score)
        index = np.where(all_score == Select_PIAA_score)[0]
        a = random.randint(0, len(index) - 1)
        b = random.randint(0, len(index) - 1)

        return all_data[index[a]], all_data[index[b]]

    def contrast_batch(self, Select_PIAA_score=5, batch_idx=0):
        all_train_users = os.listdir(self.data_dir)
        user_data = {}
        for user in all_train_users:
            user = user[:-4]
            user_data[user] = []
            image = self.peer_random_select(user, Select_PIAA_score)
            user_data[user].append(image[0])
            user_data[user].append(image[1])
        contrast_data = []
        for i in range(2):
            for key in user_data.keys():
                contrast_data.append(user_data[key][i])

        csv_save_path = os.path.join(self.save_dir, 'Contrast' + str(batch_idx) + '.csv')
        pd.DataFrame(contrast_data).to_csv(csv_save_path, sep=',', index=False)

        return 0

    def all_contrast_data(self):
        all_index = [i for i in range(100)]
        all_score = [random.randint(1, 5) for i in range(100)]
        for i in range(100):
            self.contrast_batch(all_score[i], all_index[i])

        return 0

    def read_contrast_data(self, batch_idx=0):
        csv_name = os.path.join(self.save_dir, 'Contrast' + str(batch_idx) + '.csv')
        data_image_dir = os.path.join(r'./FlickrAES_Image')
        transformed_dataset_train = FlickrDataset_MTCL(
            csv_file=csv_name,
            root_dir=data_image_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                                         ),
                ]
            )
        )
        data_train = DataLoader(transformed_dataset_train, batch_size=208,
                                shuffle=False, num_workers=3, collate_fn=my_collate, drop_last=False)

        return data_train


class Contrast_model(nn.Module):
    def __init__(self, encoder, in_dim=2048, hidden_dim=2048, out_dim=2048):
        super(Contrast_model, self).__init__()

        self.encoder = encoder
        self.predictor = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                       nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, out_dim))

    def InfoNCE(self, q1, q2, k1, k2, batch_size, T):
        l_pos1 = torch.einsum('nc,nc->n', [q1, k2]).unsqueeze(-1)  # N*1
        l_pos2 = torch.einsum('nc,nc->n', [q2, k1]).unsqueeze(-1)
        logits1 = torch.zeros((batch_size, 2 * batch_size - 1)).cuda()
        logits2 = torch.zeros((batch_size, 2 * batch_size - 1)).cuda()
        for i in range(batch_size):
            k_neg1 = torch.cat((q1[0:i, :], q1[i + 1:, :], k2[0:i, :], k2[i + 1:, :]))
            k_neg2 = torch.cat((q2[0:i, :], q2[i + 1:, :], k1[0:i, :], k1[i + 1:, :]))

            i_neg1 = torch.einsum('c,ck->k', [q1[i], torch.t(k_neg1)])  # 1*(2N-2)
            i_neg2 = torch.einsum('c,ck->k', [q2[i], torch.t(k_neg2)])  # 1*(2N-2)

            logits1[i] = (torch.cat([torch.unsqueeze(l_pos1[i], 0), torch.unsqueeze(i_neg1, 0)], dim=1))
            logits2[i] = (torch.cat([torch.unsqueeze(l_pos2[i], 0), torch.unsqueeze(i_neg2, 0)], dim=1))
        logits1 /= T
        logits2 /= T
        labels = torch.zeros(l_pos1.shape[0], dtype=torch.long).cuda()
        return logits1, logits2, labels

    def forward(self, x1, x2, batch_size, T):
        feature1, feature2 = self.encoder(x1), self.encoder(x2)
        q1, q2 = self.predictor(feature1), self.predictor(feature2)
        k1 = q1
        k2 = q2
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        k1 = nn.functional.normalize(k1, dim=1)
        k2 = nn.functional.normalize(k2, dim=1)
        logits1, logits2, labels = self.InfoNCE(q1, q2, k1, k2, batch_size, T)

        return logits1, logits2, labels


def train_contrast():
    # parameter setting
    num_epochs = 200
    best_loss = float('inf')
    T = 0.07

    # model
    backbone = models.resnet50(pretrained=True)
    giaa_model = GIAA_model(backbone)
    giaa_model.load_state_dict(
        torch.load('../model/ResNet-50/ResNet50-FlickrAes-GIAA.pt')
    )
    model = Contrast_model(giaa_model.backbone)
    model.cuda()

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=5E-2)

    # data
    data = Contrast_Database()
    data.all_contrast_data()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(100):
            data_train = data.read_contrast_data(i)
            print('***********************train***********************')
            model.train()
            loop = tqdm(enumerate(data_train), total=len(data_train), leave=True)
            for batch_idx, data in loop:
                inputs = data['image']
                PIAA_labels = data['label'] / 5.0

                batch_size = int(inputs.size()[0] / 2)
                if use_gpu:
                    inputs, PIAA_labels = Variable(inputs.float().cuda()), Variable(PIAA_labels.float().cuda())

                optimizer.zero_grad()
                logits1, logits2, labels, = model(inputs[0:batch_size], inputs[batch_size:], batch_size, T)
                criterion = nn.CrossEntropyLoss().cuda()
                loss1 = criterion(logits1, labels)
                loss2 = criterion(logits2, labels)
                loss = loss1 / 2 + loss2 / 2
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]--train')
                loop.set_postfix(loss=loss.item())

        epoch_loss = epoch_loss / 100

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            print(best_loss)
            best_model = copy.deepcopy(model.cuda())
            torch.save(best_model.cuda(), '../model/ResNet-50/ResNet50-FlickrAes-Contrast.pt')


if __name__ == '__main__':
    train_contrast()
