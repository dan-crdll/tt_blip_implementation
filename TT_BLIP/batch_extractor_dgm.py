import sys
sys.path.append('.')
from torch.utils.data import DataLoader
from TT_BLIP.data_preprocessor import DataPreprocessor
import torch
from PIL import Image
from datasets import load_dataset



class DatasetLoader:
    def __init__(self, allowed_splits=['washington_post', 'bbc', 'guardian', 'usa_today', 'simswap', 'StyleCLIP', 'HFGI', 'infoswap'], batch_size=8):
        self.dp = DataPreprocessor()
        self.allowed_splits = allowed_splits
        self.train_dataset, self.test_dataset = self.create_datasets()
        self.batch_size = batch_size

    def create_datasets(self):
        train_dataset = []
        test_dataset = []
        ds = load_dataset("rshaojimmy/DGM4", split='train')

        for el in ds:
            if (el['image'].split('/')[2] in self.allowed_splits):
                train_dataset.append({
                    'text': el['text'],
                    'image': el['image'],
                    'fake_cls': el['fake_cls']
                })
            
        ds = load_dataset("rshaojimmy/DGM4", split='validation')

        for el in ds:
            if (el['image'].split('/')[2] in self.allowed_splits):
                text_without_stopwords = ' '.join([word for word in el['text'].split() if word.lower() not in self.dp.stopwords])
                test_dataset.append({
                    'text': text_without_stopwords,
                    'image': el['image'],
                    'fake_cls': el['fake_cls']
                })
        return train_dataset, test_dataset

    def collate_fn(self, batch):
        images = []
        texts = []
        labels = []
        multi_labels = []

        poss = {
            'face_attribute': 0,
            'face_swap': 1,
            'text_attribute': 2,
            'text_swap': 3,
        }

        for b in batch:
            multi = [0, 0, 0, 0]
            path_img = f"./data/{b['image']}"
            images.append(Image.open(path_img).convert('RGB'))
            texts.append(b['text'])
            # labels.append(1 if b['fake_cls'] == 'orig' else 0)
            if b['fake_cls'] == 'orig':
                labels.append(1)
            else:
                labels.append(0)
                manip = b['fake_cls'].split('&')
                for m in manip:
                    multi[poss[m]] = 1.0
            multi_labels.append(torch.tensor(multi).unsqueeze(0))

        x = self.dp(images, texts)
        labels = torch.tensor(labels)
        multi_labels = torch.vstack(multi_labels)
        y = (labels.to(torch.float), multi_labels)
        return x, y

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, self.batch_size, collate_fn=self.collate_fn, drop_last=True)
        test_loader = DataLoader(self.test_dataset, self.batch_size, collate_fn=self.collate_fn, drop_last=True)
        return train_loader, test_loader
