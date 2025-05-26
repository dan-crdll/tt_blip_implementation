from model.version_modular.layers.feature_extraction import FeatureExtraction
from model.version_modular.layers.cross_attention_block import CrossAttnBlock
from model.version_modular.architecture import Model
from torch import nn 
from model.version_3.utils.load_data import DatasetLoader
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm 
from torch.nn import functional as F
import gc 


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_difficulty_dataset(BSZ):
    seed_everything()

    print("Building difficulty dataset")

    # PRETRAINED MODEL LOADING
    feature_extraction = FeatureExtraction(
        hf_repo_vit='google/vit-base-patch16-224',
        hf_repo_txt='microsoft/deberta-v3-base',
        unfrozen_vit=10,
        unfrozen_txt=10,
        large_vit=False,
        temp=0.05,
        queue_size=2048,
        momentum=0.999
    )

    fusion_layer = nn.ModuleList([CrossAttnBlock(768, 8, 512, 0.1) for _ in range(6)])

    def make_classifier(hidden_size, output_size, depth):
        layers = [nn.Linear(768, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        return nn.Sequential(*layers)

    bin_classifier = make_classifier(1024, 1, 3)
    multi_classifier = make_classifier(1024, 4, 3)  # reduced depth from 1024 to 3 for speed

    model = Model(feature_extraction, fusion_layer, bin_classifier, multi_classifier, 1e-4)
    model.load_from_checkpoint("./Thesis_New/wp46l1i8/checkpoints/epoch=7-step=1632.ckpt")
    model.eval()
    model.cuda()  # move to GPU if available

    # DATASET LOADING
    origins = ['washington_post', 'bbc', 'usa_today', 'guardian']
    manipulations = ['simswap', 'StyleCLIP', 'infoswap', 'HFGI']

    ds_loader = DatasetLoader(origins + manipulations, BSZ)
    train_ds = ds_loader.train_dataset
    collate_fn = ds_loader.collate_fn

    dl = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Difficulty Assessments
    processed_data = []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dl), total=len(train_ds)):
            img, txt, (y_bin, y_multi), orig = batch
            img, txt, y_bin, y_multi = img.cuda(), txt.cuda(), y_bin.cuda(), y_multi.cuda()

            (pred_bin, pred_multi), _, _ = model(img, txt, orig, y_multi, 'Val')

            loss_bin = F.binary_cross_entropy_with_logits(pred_bin, y_bin.float())
            loss_multi = F.binary_cross_entropy_with_logits(pred_multi, y_multi.float(), reduction='none')

            pred_bin = torch.sigmoid(pred_bin)
            pred_multi = torch.sigmoid(pred_multi)

            diff_bin = 0.5 * (1 - pred_bin) + 0.5 * loss_bin
            diff_mul = (0.5 * (1 - pred_multi) + 0.5 * loss_multi).mean()
            difficulty = (diff_bin + diff_mul).item()

            item = train_ds[idx]
            processed_data.append({
                'text': item['text'],
                'image': item['image'],
                'fake_cls': item['fake_cls'],
                'orig_image': item['orig_image'],
                'orig_text': item['orig_text'],
                'difficulty': difficulty
            })

    # Normalize difficulties
    difficulties = [item['difficulty'] for item in processed_data]
    min_d, max_d = min(difficulties), max(difficulties)
    print(f"Difficuties (max, min, mean): ({max_d}, {min_d}, {sum(difficulties) / len(difficulties)})")

    range_d = max_d - min_d if max_d > min_d else 1.0

    for item in processed_data:
        item['difficulty'] = (item['difficulty'] - min_d) / range_d

    class EpochTracker:
        def __init__(self):
            self.epoch = 0
        def get(self): return self.epoch
        def set(self, val): self.epoch = val

    class DifficultyDataset(IterableDataset):
        def __init__(self, data, epoch_tracker):
            self.data = data
            self.epoch_tracker = epoch_tracker

        def __iter__(self):
            epoch = self.epoch_tracker.get()
            max_d = self.difficulty_fn(epoch)
            return (sample for sample in self.data if sample['difficulty'] <= max_d)

        def difficulty_fn(self, epoch):
            if epoch < 2: return 0.33
            if epoch < 4: return 0.66
            return 1.0

    et = EpochTracker()
    dataset = DifficultyDataset(processed_data, epoch_tracker=et)
    ds_loader.train_dataset = dataset
    train_dl, val_dl = ds_loader.get_dataloaders()

    del model, img, txt, y_bin, y_multi, pred_bin, pred_multi, loss_bin, loss_multi
    torch.cuda.empty_cache()      # Clears cached memory held by allocator
    torch.cuda.ipc_collect()      # Releases memory held by inter-process communication
    gc.collect()                  # Python garbage collection

    return train_dl, val_dl, et
