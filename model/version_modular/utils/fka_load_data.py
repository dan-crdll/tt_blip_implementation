import torch
from torch.utils.data import DataLoader, IterableDataset
from model.utils.data_preprocessor import DataPreprocessor
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, ImageEnhance, ImageOps
from datasets import load_dataset
import random
import json
import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist

class FlexibleDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, drop_last=True):
        # Check if distributed training is both available AND initialized
        if dist.is_available() and dist.is_initialized():
            if num_replicas is None:
                num_replicas = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()
            self.distributed = True
        else:
            # Fall back to single-process training
            if num_replicas is None:
                num_replicas = 1
            if rank is None:
                rank = 0
            self.distributed = False
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(n, generator=g).tolist()

        # Evenly divide the indices
        if self.drop_last:
            total_size = n - (n % self.num_replicas)
        else:
            total_size = ((n + self.num_replicas - 1) // self.num_replicas) * self.num_replicas
            # Pad with duplicate indices if needed
            padding_size = total_size - n
            if padding_size > 0:
                indices += indices[:padding_size]

        indices = indices[self.rank:total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.num_replicas
        else:
            return (len(self.dataset) + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


class EpochTracker:
    def __init__(self):
        self.epoch = 0
    
    def get(self):
        return self.epoch
    
    def set(self, epoch):
        self.epoch = epoch

class DatasetLoader:
    def __init__(
        self,
        data_folder,
        allowed_splits=['washington_post', 'bbc', 'guardian', 'usa_today', 'simswap', 'StyleCLIP', 'HFGI', 'infoswap'],
        batch_size=8,
    ):
        self.dp = DataPreprocessor()
        self.allowed_splits = set(allowed_splits)
        self.batch_size = batch_size
        self.data_folder = data_folder

        # Load datasets once
        self.real_pairs_ds = load_dataset("twelcone/VisualNews")
        self.real_pairs_lookup = self._build_real_pairs_dict()

        # Load local JSONL files
        self.dgm4_train = self._load_jsonl(f"{data_folder}/train.jsonl")
        self.dgm4_val = self._load_jsonl(f"{data_folder}/val.jsonl")
        self.dgm4_test = self._load_jsonl(f"{data_folder}/test.jsonl")

        # Create datasets
        self.train_dataset = self._create_dataset(self.dgm4_train)
        self.test_dataset = self._create_dataset(self.dgm4_test, is_val=True)

    def _load_jsonl(self, file_path):
        """Load data from JSONL file"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _build_real_pairs_dict(self):
        # Build a fast lookup dictionary for id -> (img, txt)
        lookup = {}
        for split_name in ['train', 'validation', 'test']:
            split = self.real_pairs_ds[split_name]
            for item in split:
                lookup[int(item['id'])] = (item['image_path'], item['caption'])
        return lookup



    def _create_dataset(self, ds, is_val=False):
        # Filter data based on allowed splits
        filtered_data = [el for el in ds if el['image'].split('/')[2] in self.allowed_splits]

        # Efficiently create the dataset list with a list comprehension
        data_list = []
        for el in filtered_data:
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

    def augment_image_pil(self, image):
        # Random horizontal flip
        if random.random() < 0.5:
            image = ImageOps.mirror(image)

        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            image = image.rotate(angle, expand=True)

        # Random color jitter
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() < 0.5:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.8, 2.0))

        # Random crop with padding
        if random.random() < 0.5:
            padding = 10
            image = ImageOps.expand(image, border=padding, fill=0)

            left = random.randint(0, padding * 2)
            upper = random.randint(0, padding * 2)

            right = left + image.size[0] - padding * 2
            lower = upper + image.size[1] - padding * 2

            image = image.crop((left, upper, right, lower))
        
        return image

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
                labels.append(0)
            else:
                labels.append(1)
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

    def collate_fn_aug(self, batch):
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
            img = Image.open(path_img).convert('RGB')
            img = self.augment_image_pil(img)
            images.append(img)
            texts.append(b['text'])
            if b['fake_cls'] == 'orig':
                labels.append(0)
            else:
                labels.append(1)
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
        # Create train sampler
        train_sampler = FlexibleDistributedSampler(
            self.train_dataset,
            shuffle=True,
            drop_last=True
        )

        train_loader = DataLoader(
            self.train_dataset, 
            self.batch_size, 
            sampler=train_sampler,
            collate_fn=self.collate_fn_aug,
            drop_last=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=True
        )

        return train_loader, test_loader