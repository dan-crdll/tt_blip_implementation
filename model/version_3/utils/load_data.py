import torch
from torch.utils.data import DataLoader
from model.utils.data_preprocessor import DataPreprocessor
from PIL import Image
from datasets import load_dataset

class DatasetLoader:
    def __init__(
        self,
        allowed_splits=['washington_post', 'bbc', 'guardian', 'usa_today', 'simswap', 'StyleCLIP', 'HFGI', 'infoswap'],
        batch_size=8
    ):
        self.dp = DataPreprocessor()
        self.allowed_splits = set(allowed_splits)
        self.batch_size = batch_size

        # Load datasets once
        self.real_pairs_ds = load_dataset("twelcone/VisualNews")
        self.real_pairs_lookup = self._build_real_pairs_dict()

        self.dgm4_train = load_dataset("rshaojimmy/DGM4", split='train')
        self.dgm4_val = load_dataset("rshaojimmy/DGM4", split='validation')

        self.train_dataset = self._create_dataset(self.dgm4_train)
        self.test_dataset = self._create_dataset(self.dgm4_val, is_val=True)

    def _build_real_pairs_dict(self):
        # Build a fast lookup dictionary for id -> (img, txt)
        lookup = {}
        for split_name in ['train', 'validation', 'test']:
            split = self.real_pairs_ds[split_name]
            for item in split:
                lookup[int(item['id'])] = (item['image_path'], item['caption'])
        return lookup

    def _create_dataset(self, ds, is_val=False):
        # Vectorize filtering
        ds = ds.filter(lambda e: e['image'].split('/')[2] in self.allowed_splits, num_proc=4)

        # Efficiently create the dataset list with a list comprehension
        data_list = []
        for el in ds:
            id_int = int(el['id'])
            if id_int not in self.real_pairs_lookup:
                continue
            orig_img, orig_txt = self.real_pairs_lookup[id_int]
            
            orig_img = orig_img.replace('.', '', count=1).replace('/images', '')
            item = {
                'text': el['text'],
                'image': el['image'],
                'fake_cls': el['fake_cls'],
                'orig_image': orig_img,
                'orig_text': orig_txt,
            }
            data_list.append(item)
        return data_list

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
            path_img = f"./data/DGM4/origin{b['orig_image']}"
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