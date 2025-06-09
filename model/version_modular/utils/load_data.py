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
        allowed_splits=['washington_post', 'bbc', 'guardian', 'usa_today', 'simswap', 'StyleCLIP', 'HFGI', 'infoswap'],
        batch_size=8,
        difficulty=False,
    ):
        self.dp = DataPreprocessor()
        self.allowed_splits = set(allowed_splits)
        self.batch_size = batch_size

        # Load datasets once
        self.real_pairs_ds = load_dataset("twelcone/VisualNews")
        self.real_pairs_lookup = self._build_real_pairs_dict()

        if not difficulty:
            self.dgm4_train = load_dataset("rshaojimmy/DGM4", split='train')
        self.dgm4_val = load_dataset("rshaojimmy/DGM4", split='validation')

        if not difficulty:
            self.train_dataset = self._create_dataset(self.dgm4_train)
            self.et = None
        else:
            self.et = EpochTracker()
            self.train_dataset = self._create_difficulty_dataset()
        self.test_dataset = self._create_dataset(self.dgm4_val, is_val=True)

    def _build_real_pairs_dict(self):
        # Build a fast lookup dictionary for id -> (img, txt)
        lookup = {}
        for split_name in ['train', 'validation', 'test']:
            split = self.real_pairs_ds[split_name]
            for item in split:
                lookup[int(item['id'])] = (item['image_path'], item['caption'])
        return lookup

    def _create_difficulty_dataset(self):
        path = "./difficulty_dataset.json"

        with open(path, 'r') as f:
            data = json.load(f)
        
        class DifficultyDataset(torch.utils.data.Dataset):
            def __init__(self, data, epoch_tracker):
                self.data = data
                self.epoch_tracker = epoch_tracker
                self.filtered_data = self._filter()

            def _filter(self):
                epoch = self.epoch_tracker.get()
                max_d = self.difficulty_fn(epoch)
                print(f"Epoch {epoch}: Max difficulty: {max_d}")
                filtered = [sample for sample in self.data if sample['difficulty'] <= max_d]
                print(f"Epoch {epoch}: Filtered dataset size: {len(filtered)}")
                return filtered

            def difficulty_fn(self, epoch):
                if epoch == 0: return 0.66
                # if epoch < 2: return 1
                return 1.0

            def __getitem__(self, idx):
                return self.filtered_data[idx]

            def __len__(self):
                return len(self.filtered_data)

            def update_epoch(self):
                old_size = len(self.filtered_data)
                self.filtered_data = self._filter()
                new_size = len(self.filtered_data)
                print(f"Dataset updated: {old_size} -> {new_size} samples")
        
        return DifficultyDataset(data, self.et)

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
            if el['fake_cls'] == 'orig':
                bbox = torch.zeros(1, 4).float()
            else:
                if len(el['fake_image_box']) == 0:
                    bbox = torch.zeros(1, 4).float()
                else:
                    bbox = torch.tensor(el['fake_image_box']).unsqueeze(0).float()
        
            item = {
                'text': el['text'],
                'image': el['image'],
                'fake_cls': el['fake_cls'],
                'orig_image': orig_img,
                'orig_text': orig_txt,
                'bbox':bbox
            }
            data_list.append(item)
        return data_list

    def augment_image_pil(self, image, bbox=None):
        width, height = image.size
        x1, y1, x2, y2 = bbox[0]  # Assumes bbox is 1x4 tensor

        # Random horizontal flip
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            x1, x2 = width - x2, width - x1

        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            image = image.rotate(angle, expand=True)

            # Convert bbox to center format and rotate
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Translate center to origin
            cx0 = cx - width / 2
            cy0 = cy - height / 2

            # Apply rotation
            rad = math.radians(angle)
            cx_rot = cx0 * math.cos(rad) - cy0 * math.sin(rad)
            cy_rot = cx0 * math.sin(rad) + cy0 * math.cos(rad)

            # Translate back
            new_cx = cx_rot + image.size[0] / 2
            new_cy = cy_rot + image.size[1] / 2

            # Compute new bbox
            x1 = new_cx - w / 2
            y1 = new_cy - h / 2
            x2 = new_cx + w / 2
            y2 = new_cy + h / 2

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

            # Adjust bbox
            x1 -= left
            x2 -= left
            y1 -= upper
            y2 -= upper
        
        bbox = torch.tensor([[x1, y1, x2, y2]])
        # Normalize bbox coordinates with respect to image width and height
        bbox[:, [0, 2]] = bbox[:, [0, 2]] / image.size[0]
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / image.size[1]
        return image #, bbox

    def collate_fn(self, batch):
        images = []
        texts = []
        labels = []
        multi_labels = []
        original_images = []
        original_txts = []
        bboxes = []

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
            
            # Normalize bbox coordinates with respect to image width and height
            bbox = b['bbox'].clone()
            bbox[:, [0, 2]] = bbox[:, [0, 2]] / images[-1].width
            bbox[:, [1, 3]] = bbox[:, [1, 3]] / images[-1].height
            bboxes.append(bbox)
            
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
        y = (labels.to(torch.float), multi_labels, torch.vstack(bboxes))
        return images, texts, y, (original_images, original_txts)

    def collate_fn_aug(self, batch):
        images = []
        texts = []
        labels = []
        multi_labels = []
        original_images = []
        original_txts = []
        bboxes = []

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
            bboxes.append(bbox)
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
        y = (labels.to(torch.float), multi_labels, torch.vstack(bboxes))
        return images, texts, y, (original_images, original_txts)

    def get_dataloaders(self):
        # Create new sampler each time to ensure proper epoch handling
        train_sampler = FlexibleDistributedSampler(
            self.train_dataset,
            shuffle=True,
            drop_last=True
        )
        
        # Set the current epoch for the sampler
        if hasattr(self.et, 'get'):
            train_sampler.set_epoch(self.et.get())

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
