import torch 
from nltk.corpus import stopwords
import nltk
import sys
sys.path.append('.')
from torch.utils.data import DataLoader
from model.utils.data_preprocessor import DataPreprocessor
from PIL import Image
from datasets import load_dataset



class DatasetLoader:
    def __init__(self, allowed_splits=['washington_post', 'bbc', 'guardian', 'usa_today', 'simswap', 'StyleCLIP', 'HFGI', 'infoswap'], batch_size=8):
        self.dp = DataPreprocessor()
        self.allowed_splits = allowed_splits
        self.train_dataset, self.test_dataset = self.create_datasets()
        self.batch_size = batch_size

        self.real_pairs_ds = load_dataset("twelcone/VisualNews")

    def find_real_pairs(self, id):
        real_pairs_ds_train = self.real_pairs_ds['train']
        real_pairs_ds_val = self.real_pairs_ds['validation']
        real_pairs_ds_test = self.real_pairs_ds['test']

        for split in [real_pairs_ds_train, real_pairs_ds_test, real_pairs_ds_val]:
            filtered = split.filter(lambda e : e['id'] == id)

            if filtered.num_rows > 0:
                orig_txt = filtered[0]['caption']
                orig_img = filtered[0]['image_path']
                return orig_img, orig_txt
                

    def create_datasets(self):
        train_dataset = []
        test_dataset = []
        ds = load_dataset("rshaojimmy/DGM4", split='train')

        for el in ds:
            if (el['image'].split('/')[2] in self.allowed_splits):
                orig_img, orig_txt = self.find_real_pairs(int(el['id']))
                train_dataset.append({
                    'text': el['text'],
                    'image': el['image'],
                    'fake_cls': el['fake_cls'],
                    'orig_image': orig_img,
                    'orig_text': orig_txt,
                })
            
        ds = load_dataset("rshaojimmy/DGM4", split='validation')

        for el in ds:
            if (el['image'].split('/')[2] in self.allowed_splits):
                orig_img, orig_txt = self.find_real_pairs(int(el['id']))
                test_dataset.append({
                    'text': el['text'],
                    'image': el['image'],
                    'fake_cls': el['fake_cls'],
                    'orig_image': orig_img.replace('.', ''),
                    'orig_text': orig_txt,
                })
        return train_dataset, test_dataset

    def collate_fn(self, batch):
        images = []
        texts = []
        labels = []
        multi_labels = []

        original_images = []
        original_txts = []

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
            
            if b['fake_cls'] == 'orig':
                labels.append(1)
            else:
                labels.append(0)
                manip = b['fake_cls'].split('&')
                for m in manip:
                    multi[poss[m]] = 1.0
            multi_labels.append(torch.tensor(multi).unsqueeze(0))

            path_img = f"./data{b['orig_image']}"
            original_images.append(Image.open(path_img).convert('RGB'))
            original_txts.append(b['orig_text'])

        labels = torch.tensor(labels)
        multi_labels = torch.vstack(multi_labels)
        y = (labels.to(torch.float), multi_labels)
        return images, texts, y, (original_images, original_txts)

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, self.batch_size, collate_fn=self.collate_fn, drop_last=True)
        test_loader = DataLoader(self.test_dataset, self.batch_size, collate_fn=self.collate_fn, drop_last=True)
        return train_loader, test_loader
