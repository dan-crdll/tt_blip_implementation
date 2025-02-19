import sys
sys.path.append('.')
from torch.utils.data import DataLoader
from TT_BLIP.data_preprocessor import DataPreprocessor
import os 
import torch
from PIL import Image
import json
from datasets import load_dataset

class DatasetLoader:
    def __init__(self, batch_size=8):
        self.dp = DataPreprocessor()
        self.train_dataset, self.test_dataset = self.create_datasets()

    def create_datasets(self):
        train_dataset = load_dataset("rshaojimmy/DGM4", split='train')
        test_dataset = load_dataset("rshaojimmy/DGM4", split='validation')
        return train_dataset, test_dataset

    def collate_fn(self, batch):
        images = []
        texts = []
        labels = []

        for b in batch:
            path_img = f"./data/{b['image']}"
            images.append(Image.open(path_img).convert('RGB'))
            texts.append(b['text'])
            labels.append(1 if b['fake_cls'] == 'orig' else 0)

        x = self.dp(images, texts)
        y = labels.to(torch.float)
        return x, y

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, self.batch_size, collate_fn=self.collate_fn)
        test_loader = DataLoader(self.test_dataset, self.batch_size, collate_fn=self.collate_fn)
        return train_loader, test_loader