import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model.version_modular.layers.feature_extraction import FeatureExtraction
from model.version_modular.layers.cross_attention_block import CrossAttnBlock
from model.version_modular.architecture import Model
from model.version_3.utils.load_data import DatasetLoader


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_classifier(hidden_size, output_size, depth):
    layers = [nn.Linear(768, hidden_size), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)


def create_difficulty_dataset(BSZ=8, tmp_path="difficulty_temp.jsonl", final_path="difficulty_dataset.json"):
    seed_everything()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Feature extractor & classifiers
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
    bin_classifier = make_classifier(1024, 1, 3)
    multi_classifier = make_classifier(1024, 4, 3)

    model = Model(feature_extraction, fusion_layer, bin_classifier, multi_classifier, 1e-4)
    model.load_partial_weights("./Thesis_New/wp46l1i8/checkpoints/epoch=7-step=1632.ckpt")
    model = model.cuda()
    model.eval()

    # Dataset
    origins = ['washington_post', 'bbc', 'usa_today', 'guardian']
    manipulations = ['simswap', 'StyleCLIP', 'infoswap', 'HFGI']
    ds_loader = DatasetLoader(origins + manipulations, BSZ)
    train_ds = ds_loader.train_dataset
    collate_fn = ds_loader.collate_fn
    dl = DataLoader(train_ds, batch_size=BSZ, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Difficulty generation (1st pass)
    current_idx = 0
    min_d, max_d = float('inf'), float('-inf')

    with torch.no_grad(), open(tmp_path, 'w', encoding='utf-8') as f:
        for batch in tqdm(dl, total=len(train_ds) // BSZ):
            img, txt, (y_bin, y_multi), orig = batch
            img, txt, y_bin, y_multi = img, txt, y_bin, y_multi

            (pred_bin, pred_multi), _, _ = model(img, txt, orig, y_multi, 'Val')
            pred_bin = pred_bin.cpu()
            pred_multi = pred_multi.cpu()
            y_bin = y_bin.cpu()
            y_multi = y_multi.cpu()
            loss_bin = F.binary_cross_entropy_with_logits(pred_bin, y_bin.float(), reduction='none')
            loss_multi = F.binary_cross_entropy_with_logits(pred_multi, y_multi.float(), reduction='none')
            pred_bin_sig = torch.sigmoid(pred_bin)
            pred_multi_sig = torch.sigmoid(pred_multi)

            diff_bin = 0.5 * (1 - pred_bin_sig.squeeze()) + 0.5 * loss_bin
            diff_mul = (0.5 * (1 - pred_multi_sig) + 0.5 * loss_multi).mean(dim=1)
            difficulties = (diff_bin + diff_mul).tolist()

            for i in range(len(difficulties)):
                if current_idx + i >= len(train_ds): break
                item = train_ds[current_idx + i]
                record = {
                    'text': item['text'],
                    'image': item['image'],
                    'fake_cls': item['fake_cls'],
                    'orig_image': item['orig_image'],
                    'orig_text': item['orig_text'],
                    'difficulty': difficulties[i]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                min_d = min(min_d, difficulties[i])
                max_d = max(max_d, difficulties[i])
            current_idx += BSZ

    range_d = max_d - min_d if max_d > min_d else 1.0
    print(f"Difficulty (max, min): ({max_d}, {min_d})")

    # Normalization (2nd pass)
    with open(tmp_path, 'r', encoding='utf-8') as fin, open(final_path, 'w', encoding='utf-8') as fout:
        data = []
        for line in fin:
            item = json.loads(line)
            item['difficulty'] = (item['difficulty'] - min_d) / range_d
            data.append(item)
        json.dump(data, fout, ensure_ascii=False, indent=2)

    print(f"Final normalized difficulty dataset saved to: {final_path}")
    os.remove(tmp_path)

if __name__ == "__main__":
    create_difficulty_dataset()
