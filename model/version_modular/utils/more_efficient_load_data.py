import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from model.utils.data_preprocessor import DataPreprocessor
from PIL import Image, ImageEnhance, ImageOps
from datasets import load_dataset
import random
import json
import math
import torch.distributed as dist
from functools import lru_cache
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from threading import Lock
import warnings
warnings.filterwarnings('ignore')

class FlexibleDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, drop_last=True):
        if dist.is_available() and dist.is_initialized():
            if num_replicas is None:
                num_replicas = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()
            self.distributed = True
        else:
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
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.drop_last:
            total_size = n - (n % self.num_replicas)
        else:
            total_size = ((n + self.num_replicas - 1) // self.num_replicas) * self.num_replicas
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
        self._lock = Lock()
    
    def get(self):
        with self._lock:
            return self.epoch
    
    def set(self, epoch):
        with self._lock:
            self.epoch = epoch


class OptimizedDataset(Dataset):
    """Optimized dataset with lazy loading and caching"""
    
    def __init__(self, data_list, real_pairs_lookup, is_val=False, cache_size=1000):
        self.data_list = data_list
        self.real_pairs_lookup = real_pairs_lookup
        self.is_val = is_val
        self.cache_size = cache_size
        
        # Position mapping for multi-labels (define before precompute)
        self.poss = {
            'face_attribute': 0,
            'face_swap': 1,
            'text_attribute': 2,
            'text_swap': 3,
        }
        
        # Pre-compute static data
        self._precompute_labels()
        
        # Image cache with LRU eviction
        self._image_cache = {}
        self._cache_order = []
        self._cache_lock = Lock()

    def _precompute_labels(self):
        """Pre-compute labels and multi-labels to avoid repeated computation"""
        self.labels = []
        self.multi_labels = []
        self.bboxes = []
        
        for item in self.data_list:
            # Binary label
            label = 0 if item['fake_cls'] == 'orig' else 1
            self.labels.append(label)
            
            # Multi-label
            multi = [0, 0, 0, 0]
            if item['fake_cls'] != 'orig':
                manip = item['fake_cls'].split('&')
                for m in manip:
                    if m in self.poss:
                        multi[self.poss[m]] = 1.0
            self.multi_labels.append(torch.tensor(multi, dtype=torch.float))
            
            # Bbox (already processed)
            self.bboxes.append(item['bbox'])

    @lru_cache(maxsize=128)
    def _get_cached_image_path(self, image_path, is_original=False):
        """Cache image path construction"""
        if is_original:
            return f"./data/DGM4/origin{image_path}"
        return f"./data/{image_path}"

    def _load_image_with_cache(self, path):
        """Load image with caching"""
        with self._cache_lock:
            if path in self._image_cache:
                # Move to end (most recently used)
                self._cache_order.remove(path)
                self._cache_order.append(path)
                return self._image_cache[path]
            
            # Load image
            try:
                img = Image.open(path).convert('RGB')
                
                # Cache management
                if len(self._image_cache) >= self.cache_size:
                    # Remove least recently used
                    lru_path = self._cache_order.pop(0)
                    del self._image_cache[lru_path]
                
                self._image_cache[path] = img
                self._cache_order.append(path)
                return img
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Return a dummy image to prevent crashes
                return Image.new('RGB', (224, 224), color='black')

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Get paths
        img_path = self._get_cached_image_path(item['image'])
        orig_img_path = self._get_cached_image_path(item['orig_image'], is_original=True)
        
        # Load images
        image = self._load_image_with_cache(img_path)
        orig_image = self._load_image_with_cache(orig_img_path)
        
        return {
            'image': image,
            'text': item['text'],
            'label': self.labels[idx],
            'multi_label': self.multi_labels[idx],
            'bbox': self.bboxes[idx],
            'orig_image': orig_image,
            'orig_text': item['orig_text']
        }

    def __len__(self):
        return len(self.data_list)


class DifficultyDataset(Dataset):
    """Optimized difficulty dataset"""
    
    def __init__(self, data, epoch_tracker, real_pairs_lookup):
        self.data = data
        self.epoch_tracker = epoch_tracker
        self.real_pairs_lookup = real_pairs_lookup
        self.filtered_data = self._filter()
        self._dataset = None
        self._current_epoch = -1

    def _filter(self):
        epoch = self.epoch_tracker.get()
        max_d = self.difficulty_fn(epoch)
        print(f"Epoch {epoch}: Max difficulty: {max_d}")
        filtered = [sample for sample in self.data if sample['difficulty'] <= max_d]
        print(f"Epoch {epoch}: Filtered dataset size: {len(filtered)}")
        return filtered

    def difficulty_fn(self, epoch):
        if epoch == 0: 
            return 0.66
        return 1.0

    def _ensure_dataset_updated(self):
        """Lazily update the underlying dataset if epoch changed"""
        current_epoch = self.epoch_tracker.get()
        if current_epoch != self._current_epoch:
            self.filtered_data = self._filter()
            self._dataset = OptimizedDataset(self.filtered_data, self.real_pairs_lookup)
            self._current_epoch = current_epoch

    def __getitem__(self, idx):
        self._ensure_dataset_updated()
        return self._dataset[idx]

    def __len__(self):
        self._ensure_dataset_updated()
        return len(self._dataset)

    def update_epoch(self):
        old_size = len(self.filtered_data)
        self.filtered_data = self._filter()
        new_size = len(self.filtered_data)
        print(f"Dataset updated: {old_size} -> {new_size} samples")


class DatasetLoader:
    def __init__(
        self,
        allowed_splits=['washington_post', 'bbc', 'guardian', 'usa_today', 'simswap', 'StyleCLIP', 'HFGI', 'infoswap'],
        batch_size=8,
        difficulty=False,
        num_workers=None,
        prefetch_factor=2
    ):
        self.dp = DataPreprocessor()
        self.allowed_splits = set(allowed_splits)
        self.batch_size = batch_size
        
        # Optimize number of workers
        if num_workers is None:
            self.num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead
        else:
            self.num_workers = num_workers
        
        self.prefetch_factor = prefetch_factor

        print("Loading datasets...")
        # Load datasets with optimizations
        self.real_pairs_ds = load_dataset("twelcone/VisualNews", num_proc=4)
        self.real_pairs_lookup = self._build_real_pairs_dict()

        if not difficulty:
            self.dgm4_train = load_dataset("rshaojimmy/DGM4", split='train', num_proc=4)
        self.dgm4_val = load_dataset("rshaojimmy/DGM4", split='validation', num_proc=4)

        print("Creating datasets...")
        if not difficulty:
            train_data_list = self._create_dataset_list(self.dgm4_train)
            self.train_dataset = OptimizedDataset(train_data_list, self.real_pairs_lookup)
            self.et = None
        else:
            self.et = EpochTracker()
            self.train_dataset = self._create_difficulty_dataset()
        
        test_data_list = self._create_dataset_list(self.dgm4_val, is_val=True)
        self.test_dataset = OptimizedDataset(test_data_list, self.real_pairs_lookup, is_val=True)
        print("Dataset creation complete!")

    def _build_real_pairs_dict(self):
        """Optimized lookup dictionary building"""
        print("Building real pairs lookup...")
        lookup = {}
        
        for split_name in ['train', 'validation', 'test']:
            split = self.real_pairs_ds[split_name]
            for item in split:
                lookup[int(item['id'])] = (item['image_path'], item['caption'])
        
        print(f"Built lookup with {len(lookup)} entries")
        return lookup

    def _create_difficulty_dataset(self):
        path = "./difficulty_dataset.json"
        with open(path, 'r') as f:
            data = json.load(f)
        return DifficultyDataset(data, self.et, self.real_pairs_lookup)

    def _create_dataset_list(self, ds, is_val=False):
        """Optimized dataset creation with batch processing"""
        print(f"Filtering dataset (is_val={is_val})...")
        
        # Use more processes for filtering
        ds_filtered = ds.filter(
            lambda e: e['image'].split('/')[2] in self.allowed_splits, 
            num_proc=min(mp.cpu_count(), 8),
            desc="Filtering by allowed splits"
        )
        
        print(f"Creating data list from {len(ds_filtered)} items...")
        data_list = []
        
        for el in ds_filtered:
            id_int = int(el['id'])
            if id_int not in self.real_pairs_lookup:
                continue
                
            orig_img, orig_txt = self.real_pairs_lookup[id_int]
            orig_img = orig_img.replace('.', '', count=1).replace('/images', '')
            
            # Pre-process bbox
            if el['fake_cls'] == 'orig' or len(el['fake_image_box']) == 0:
                bbox = torch.zeros(1, 4, dtype=torch.float)
            else:
                bbox = torch.tensor(el['fake_image_box'], dtype=torch.float).unsqueeze(0)
        
            item = {
                'text': el['text'],
                'image': el['image'],
                'fake_cls': el['fake_cls'],
                'orig_image': orig_img,
                'orig_text': orig_txt,
                'bbox': bbox
            }
            data_list.append(item)
        
        print(f"Created data list with {len(data_list)} items")
        return data_list

    def augment_image_fast(self, image, bbox=None):
        """Corrected image augmentation with proper bbox handling"""
        if bbox is None:
            # Se non c'è bbox, restituisci bbox vuota
            bbox = torch.zeros(1, 4)
            bbox_provided = False
        else:
            bbox_provided = True
        
        # Converti bbox in coordinate assolute se fornita
        if bbox_provided:
            orig_width, orig_height = image.size
            x1, y1, x2, y2 = bbox[0].tolist()
            
            # Se la bbox è già normalizzata (valori 0-1), denormalizza
            if max(x1, y1, x2, y2) <= 1.0:
                x1, x2 = x1 * orig_width, x2 * orig_width
                y1, y2 = y1 * orig_height, y2 * orig_height
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0

        # Batch random decisions
        flip_h = random.random() < 0.5
        do_rotate = random.random() < 0.3  # Ridotto per evitare bbox troppo distorte
        do_brightness = random.random() < 0.5
        do_contrast = random.random() < 0.5
        do_sharpness = random.random() < 0.5
        do_crop = random.random() < 0.3  # Ridotto per preservare bbox

        # 1. FLIP ORIZZONTALE
        if flip_h and bbox_provided:
            image = ImageOps.mirror(image)
            width = image.size[0]
            x1, x2 = width - x2, width - x1

        # 2. ROTAZIONE (solo se bbox fornita)
        if do_rotate and bbox_provided:
            angle = random.uniform(-10, 10)  # Range ridotto
            
            # Calcola tutti i 4 vertici della bbox
            corners = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ])
            
            # Centro dell'immagine originale
            orig_center = np.array([image.size[0]/2, image.size[1]/2])
            
            # Applica rotazione
            image = image.rotate(angle, expand=True, fillcolor=(128, 128, 128))
            
            # Calcola la nuova bbox dopo rotazione
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            
            # Matrice di rotazione
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            
            # Trasla i corner al centro, ruota, poi trasla alla nuova posizione
            centered_corners = corners - orig_center
            rotated_corners = centered_corners @ rotation_matrix.T
            
            # Nuovo centro dell'immagine espansa
            new_center = np.array([image.size[0]/2, image.size[1]/2])
            final_corners = rotated_corners + new_center
            
            # Calcola la nuova bbox che racchiude tutti i corner
            x1 = np.min(final_corners[:, 0])
            y1 = np.min(final_corners[:, 1])
            x2 = np.max(final_corners[:, 0])
            y2 = np.max(final_corners[:, 1])
            
            # Clamp alle dimensioni dell'immagine
            x1 = max(0, min(x1, image.size[0]))
            y1 = max(0, min(y1, image.size[1]))
            x2 = max(0, min(x2, image.size[0]))
            y2 = max(0, min(y2, image.size[1]))

        # 3. CROPPING
        if do_crop and bbox_provided:
            padding = min(5, image.size[0]//20, image.size[1]//20)  # Padding adattivo
            
            # Assicurati che il crop non elimini completamente la bbox
            max_left = min(padding, int(x1))
            max_upper = min(padding, int(y1))
            
            left = random.randint(0, max_left) if max_left > 0 else 0
            upper = random.randint(0, max_upper) if max_upper > 0 else 0
            
            new_width = image.size[0] - padding
            new_height = image.size[1] - padding
            
            # Assicurati che le nuove dimensioni siano valide
            if new_width > left and new_height > upper:
                image = image.crop((left, upper, left + new_width, upper + new_height))
                
                # Aggiorna bbox
                x1, x2 = x1 - left, x2 - left
                y1, y2 = y1 - upper, y2 - upper
                
                # Clamp alla nuova immagine
                x1 = max(0, min(x1, image.size[0]))
                y1 = max(0, min(y1, image.size[1]))
                x2 = max(0, min(x2, image.size[0]))
                y2 = max(0, min(y2, image.size[1]))

        # 4. TRASFORMAZIONI COLORE (non influenzano bbox)
        if do_brightness:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        if do_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        if do_sharpness:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.5, 1.5))

        # 5. NORMALIZZA BBOX
        if bbox_provided:
            # Verifica che la bbox sia ancora valida
            if x2 > x1 and y2 > y1:
                bbox_norm = torch.tensor([[
                    x1 / image.size[0], y1 / image.size[1],
                    x2 / image.size[0], y2 / image.size[1]
                ]], dtype=torch.float32)
            else:
                # Bbox invalidata dalle trasformazioni
                bbox_norm = torch.zeros(1, 4, dtype=torch.float32)
        else:
            bbox_norm = torch.zeros(1, 4, dtype=torch.float32)
        
        return image, bbox_norm


    def augment_image_robust(self, image, bbox=None):
        """More robust version with validation"""
        if bbox is None:
            return self.augment_image_fast(image, bbox)
        
        # Salva stato originale per fallback
        orig_image = image.copy()
        orig_bbox = bbox.clone()
        
        try:
            aug_image, aug_bbox = self.augment_image_fast(image, bbox)
            
            # Valida il risultato
            if (aug_bbox[0] == 0).all():  # Bbox invalidata
                return orig_image, orig_bbox
                
            # Verifica che la bbox sia ragionevole
            x1, y1, x2, y2 = aug_bbox[0].tolist()
            if x2 - x1 < 0.01 or y2 - y1 < 0.01:  # Bbox troppo piccola
                return orig_image, orig_bbox
                
            return aug_image, aug_bbox
            
        except Exception:
            # Fallback in caso di errore
            return orig_image, orig_bbox

    def augment_image_conservative(self, image, bbox=None):
        """Augmentation conservativa che preserva meglio le bbox"""
        if bbox is None:
            bbox = torch.zeros(1, 4)
            bbox_provided = False
        else:
            bbox_provided = True
    
        if bbox_provided:
            orig_width, orig_height = image.size
            x1, y1, x2, y2 = bbox[0].tolist()
            
            # Denormalizza se necessario
            if max(x1, y1, x2, y2) <= 1.0:
                x1, x2 = x1 * orig_width, x2 * orig_width
                y1, y2 = y1 * orig_height, y2 * orig_height
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0

        # Probabilità ridotte per trasformazioni aggressive
        flip_h = random.random() < 0.5
        do_rotate = random.random() < 0.2  # Molto ridotta
        do_brightness = random.random() < 0.6
        do_contrast = random.random() < 0.6
        do_sharpness = random.random() < 0.4
        do_crop = random.random() < 0.3

        # 1. FLIP ORIZZONTALE (sicuro)
        if flip_h and bbox_provided:
            image = ImageOps.mirror(image)
            width = image.size[0]
            x1, x2 = width - x2, width - x1

        # 2. ROTAZIONE LIMITATA (solo piccoli angoli)
        if do_rotate and bbox_provided:
            angle = random.uniform(-5, 5)  # Angoli molto piccoli
            
            # Usa rotazione senza expand per mantenere dimensioni
            image = image.rotate(angle, expand=False, fillcolor=(128, 128, 128))
            
            # Per angoli piccoli, approssimazione semplice è sufficiente
            center_x, center_y = image.size[0]/2, image.size[1]/2
            bbox_center_x, bbox_center_y = (x1 + x2)/2, (y1 + y2)/2
            
            # Offset dal centro dell'immagine
            offset_x, offset_y = bbox_center_x - center_x, bbox_center_y - center_y
            
            # Rotazione del centro della bbox
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            
            new_offset_x = offset_x * cos_a - offset_y * sin_a
            new_offset_y = offset_x * sin_a + offset_y * cos_a
            
            new_center_x = center_x + new_offset_x
            new_center_y = center_y + new_offset_y
            
            # Mantieni dimensioni bbox
            w, h = x2 - x1, y2 - y1
            x1 = new_center_x - w/2
            y1 = new_center_y - h/2
            x2 = new_center_x + w/2
            y2 = new_center_y + h/2
            
            # Clamp
            x1 = max(0, min(x1, image.size[0]))
            y1 = max(0, min(y1, image.size[1]))
            x2 = max(0, min(x2, image.size[0]))
            y2 = max(0, min(y2, image.size[1]))

        # 3. CROPPING MOLTO CONSERVATIVO
        if do_crop and bbox_provided:
            # Padding massimo del 2% delle dimensioni
            max_padding = min(image.size[0] * 0.02, image.size[1] * 0.02, 10)
            
            # Assicurati di non tagliare la bbox
            safe_left = min(max_padding, x1 * 0.1)  # Max 10% del margine sinistro della bbox
            safe_upper = min(max_padding, y1 * 0.1)
            
            left = random.uniform(0, safe_left)
            upper = random.uniform(0, safe_upper)
            
            right = image.size[0] - random.uniform(0, max_padding)
            bottom = image.size[1] - random.uniform(0, max_padding)
            
            # Verifica che la bbox rimanga nell'immagine
            if right > x2 and bottom > y2 and left < x1 and upper < y1:
                image = image.crop((int(left), int(upper), int(right), int(bottom)))
                
                # Aggiorna bbox
                x1, x2 = x1 - left, x2 - left
                y1, y2 = y1 - upper, y2 - upper

        # 4. TRASFORMAZIONI COLORE (conservative)
        if do_brightness:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))  # Range ridotto

        if do_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))

        if do_sharpness:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # 5. NORMALIZZA E VALIDA
        if bbox_provided:
            # Controlla che la bbox sia ancora valida
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                bbox_norm = torch.tensor([[
                    x1 / image.size[0], y1 / image.size[1],
                    x2 / image.size[0], y2 / image.size[1]
                ]], dtype=torch.float32)
                
                # Clamp tra 0 e 1
                bbox_norm = torch.clamp(bbox_norm, 0.0, 1.0)
            else:
                bbox_norm = torch.zeros(1, 4, dtype=torch.float32)
        else:
            bbox_norm = torch.zeros(1, 4, dtype=torch.float32)
    
        return image, bbox_norm

    def collate_fn(self, batch):
        """Optimized collate function"""
        images, texts, labels, multi_labels = [], [], [], []
        original_images, original_txts, bboxes = [], [], []

        for item in batch:
            images.append(item['image'])
            texts.append(item['text'])
            labels.append(item['label'])
            multi_labels.append(item['multi_label'])
            original_images.append(item['orig_image'])
            original_txts.append(item['orig_text'])
            
            # Normalize bbox
            bbox = item['bbox'].clone()
            if item['image'].size[0] > 0 and item['image'].size[1] > 0:
                bbox[:, [0, 2]] = bbox[:, [0, 2]] / item['image'].size[0]
                bbox[:, [1, 3]] = bbox[:, [1, 3]] / item['image'].size[1]
            bboxes.append(bbox)

        # Stack tensors efficiently
        labels = torch.tensor(labels, dtype=torch.float)
        multi_labels = torch.stack(multi_labels)
        bboxes = torch.cat(bboxes, dim=0)
        
        y = (labels, multi_labels)
        return images, texts, y, (original_images, original_txts)

    def collate_fn_aug(self, batch):
        """Optimized augmented collate function"""
        images, texts, labels, multi_labels = [], [], [], []
        original_images, original_txts, bboxes = [], [], []

        for item in batch:
            # Apply augmentation
            img, bbox = self.augment_image_conservative(item['image'], item['bbox'])
            
            images.append(img)
            texts.append(item['text'])
            labels.append(item['label'])
            multi_labels.append(item['multi_label'])
            original_images.append(item['orig_image'])
            original_txts.append(item['orig_text'])
            bboxes.append(bbox)

        # Stack tensors efficiently
        labels = torch.tensor(labels, dtype=torch.float)
        multi_labels = torch.stack(multi_labels)
        bboxes = torch.cat(bboxes, dim=0)
        
        y = (labels, multi_labels)
        return images, texts, y, (original_images, original_txts)

    def get_dataloaders(self):
        """Create optimized data loaders"""
        train_sampler = FlexibleDistributedSampler(
            self.train_dataset,
            shuffle=True,
            drop_last=True
        )
        
        if hasattr(self.et, 'get'):
            train_sampler.set_epoch(self.et.get())

        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            sampler=train_sampler,
            collate_fn=self.collate_fn_aug,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
            shuffle=False
        )

        return train_loader, test_loader