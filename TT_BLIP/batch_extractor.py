import sys
sys.path.append('.')
from torch.utils.data import DataLoader
from TT_BLIP.data_preprocessor import DataPreprocessor
import os 
import torch
from PIL import Image
import json

class DatasetLoader:
    def __init__(self, data_dir='./data/gossipcop', batch_size=8, train_ratio=0.8, balance=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.dp = DataPreprocessor()
        self.train_dataset, self.test_dataset = self.create_datasets(balance)

    def create_datasets(self, balance):
        if balance:
            fake_idxs = torch.arange(0, len(os.listdir(f'{self.data_dir}/real/img'))).unsqueeze(-1)
        else:
            fake_idxs = torch.arange(0, len(os.listdir(f'{self.data_dir}/fake/img'))).unsqueeze(-1)
        true_idxs = torch.arange(0, len(os.listdir(f'{self.data_dir}/real/img'))).unsqueeze(-1)

        fake_idxs = torch.cat([fake_idxs, torch.zeros_like(fake_idxs)], -1)
        true_idxs = torch.cat([true_idxs, torch.ones_like(true_idxs)], -1)

        dataset = torch.cat([fake_idxs, true_idxs])
        dataset = dataset[torch.randperm(dataset.size(0))]  # shuffling

        train_size = int(self.train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        return train_dataset, test_dataset

    def collate_fn(self, batch):
        batch = torch.vstack(batch)
        idxs, labels = batch[:, 0], batch[:, 1]

        images = []
        texts = []

        for i, idx in enumerate(idxs):
            path_img = f"{self.data_dir}/{'fake' if labels[i] == 0 else 'real'}/img/{idx}.jpg"
            path_txt = f"{self.data_dir}/{'fake' if labels[i] == 0 else 'real'}/text/{idx}.json"

            images.append(Image.open(path_img).convert('RGB'))
            with open(path_txt, 'r') as fp:
                texts.append(json.load(fp)['text'])
        x = self.dp(images, texts)
        y = labels.to(torch.float)
        return x, y

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, self.batch_size, collate_fn=self.collate_fn)
        test_loader = DataLoader(self.test_dataset, self.batch_size, collate_fn=self.collate_fn)
        return train_loader, test_loader