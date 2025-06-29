from model.version_modular.layers.feature_extraction import create_feature_extraction
from model.version_modular.layers.cross_attention_block import create_fusion_layer
from model.version_modular.efficient_architecture import Model, compute_eer
from torch import nn 
from model.version_modular.utils.fka_load_data import DatasetLoader
from lightning.pytorch.loggers import WandbLogger
import torch
from dgm4_download import download_dgm4
import yaml
import random
import os
import numpy as np
import lightning as L
from build_difficulty_dataset import create_difficulty_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm.auto import tqdm
import torchmetrics


class DGM4DataModule(L.LightningDataModule):
    def __init__(self, ds_loader, et):
        super().__init__()
        self.dataset_loader = ds_loader
        self.epoch_tracker = et
        self.current_epoch = 0

    def train_dataloader(self):
        # Update epoch tracker before creating dataloader
        self.epoch_tracker.set(self.current_epoch)
        
        # Update dataset with new epoch
        if hasattr(self.dataset_loader.train_dataset, "update_epoch"):
            self.dataset_loader.train_dataset.update_epoch()
        
        return self.dataset_loader.get_dataloaders()[0]

    def val_dataloader(self):
        return self.dataset_loader.get_dataloaders()[1]

    def on_train_epoch_start(self):
        # This gets called by the trainer at the start of each epoch
        pass

    def setup(self, stage=None):
        # Called on every process in DDP
        pass


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_classifiers(hidden_dim_bin, hidden_dim_multi, num_layers_bin, num_layers_multi):
    print("#### INITIALIZING CLASSIFIERS ####")
    bin_layers = [nn.Linear(768, hidden_dim_bin)]
    for _ in range(num_layers_bin):
        bin_layers.append(nn.Linear(hidden_dim_bin, hidden_dim_bin))
        bin_layers.append(nn.ReLU())
    bin_layers.append(nn.Linear(hidden_dim_bin, 1))

    multi_layers = [nn.Linear(768, hidden_dim_multi)]
    for _ in range(num_layers_multi):
        multi_layers.append(nn.Linear(hidden_dim_multi, hidden_dim_multi))
        multi_layers.append(nn.ReLU())
    multi_layers.append(nn.Linear(hidden_dim_multi, 4))

    return nn.Sequential(*bin_layers), nn.Sequential(*multi_layers)


def main():
    split_train = 'washington_post'
    print("##### CONFIGURATION #####")
    
    lr = 1e-4#float(input("Learning rate (e.g., 1e-3): "))
    batch_size = 32#int(input("Batch size: "))
    epochs = 20#int(input("Epochs: "))
    grad_acc = 1#int(input("Gradient accumulation: "))
    gpus_input = "0"#input("GPUs (comma-separated, no spaces): ")
    grad_clip = 1.0#float(input("Gradient clipping: "))
    curriculum = 0#int(input("Use curriculum learning: Y (1) | N (0): "))
    num_layers_bin = 3#int(input("Number of layers for binary classifier: "))
    num_layers_multi = 3#int(input("Number of layers for multi-label classifier: "))
    hidden_dim_bin = 1024#int(input("Hidden dim for binary classifier: "))
    hidden_dim_multi = 1024#int(input("Hidden dim for multi-label classifier: "))
    blip = 1#int(input("Use Blip: Y (1) | N(0): "))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_input
    gpus = [int(gpu) for gpu in gpus_input.split(",")]

    seed_everything()

    origins = ['washington_post', 'bbc', 'usa_today', 'guardian']
    manipulations = ['simswap', 'StyleCLIP', 'infoswap', 'HFGI']

    loader = DatasetLoader(
        data_folder="./metadata_split/bbc",
        batch_size=batch_size
    )
    _, bbc_dl = loader.get_dataloaders()

    loader = DatasetLoader(
        data_folder="./metadata_split/guardian",
        batch_size=batch_size
    )
    _, guardian_dl = loader.get_dataloaders()

    loader = DatasetLoader(
        data_folder="./metadata_split/usa_today",
        batch_size=batch_size
    )
    _, usa_today_dl = loader.get_dataloaders()

    loader = DatasetLoader(
        data_folder="./metadata_split/washington_post",
        batch_size=batch_size
    )
    _, washington_post_dl = loader.get_dataloaders()

    feature_extraction_layer = create_feature_extraction()
    fusion_layer = create_fusion_layer()
    bin_classifier, multi_classifier = create_classifiers(
        hidden_dim_bin, hidden_dim_multi, num_layers_bin, num_layers_multi
    )

    logger = WandbLogger('BI_DEC_DGM4', project="Thesis_New")
    torch.set_float32_matmul_precision('high')

    model = Model.load_from_checkpoint("./Thesis_New/iuwpggb3/checkpoints/washington_post_trained_model.ckpt",
        feature_extraction_layer=feature_extraction_layer,
        fusion_layer=fusion_layer,
        classifier_bin=bin_classifier,
        classifier_multi=multi_classifier,
        lr=lr,
        epoch_tracker=None, 
        use_blip=blip,
        dataModule=None,
        arcface=False
    )
    model.eval()
    device = model.device

    accuracy = [[], [], [], []]
    eer = [[], [], [], []]
    auc = [[], [], [], []]

    metrics = []
    split = ['bbc', 'guardian', 'usa_today', 'washington_post']
    for num, dl in enumerate([bbc_dl, guardian_dl, usa_today_dl, washington_post_dl]):
        print(f"SPLIT {split[num]}")
        for batch in tqdm(dl):
            img, txt, (y_bin, y_multi), orig = batch

            
            # Move only the label tensors to device (img and txt are handled by the model)
            y_bin = y_bin.to(device)
            y_multi = y_multi.to(device)
            
            # Convert binary labels to integers for metrics that expect integer targets
            y_bin_int = (y_bin > 0.5).long()  # Convert float labels to 0/1 integers

            with torch.no_grad():
                (p_bin, _), *_ = model(img, txt, orig, y_multi, "Val")

                pred_bin_sigmoid = torch.sigmoid(p_bin)

                accuracy[num].append(
                    (pred_bin_sigmoid.round() == y_bin_int).float().mean().item()
                )

                eer[num].append(
                    compute_eer(pred_bin_sigmoid, y_bin_int)
                )

                auc[num].append(
                    torchmetrics.functional.auroc(pred_bin_sigmoid, y_bin_int, task="binary").item()
                )
        
        metrics.append(
            {
                'Accuracy': np.mean(accuracy[num]),
                'EER': np.mean(eer[num]),
                'AUC': np.mean(auc[num]),
            }
        )

        print(metrics[-1])


    with open(f"fka/fka_metrics_{split_train}.txt", 'w') as file:
        for i, metric in enumerate(metrics):
            print(f"Dataset: {split[i]}")
            print(f"Accuracy: {metric['Accuracy']:.4f}")
            print(f"EER: {metric['EER']:.4f}")
            print(f"AUC: {metric['AUC']:.4f}")
            print("-" * 30)
            file.write(f"Dataset: {split[i]}\n")
            file.write(f"Accuracy: {metric['Accuracy']:.4f}\n")
            file.write(f"EER: {metric['EER']:.4f}\n")
            file.write(f"AUC: {metric['AUC']:.4f}\n")
            file.write("-" * 30 + "\n")

if __name__ == "__main__":
    main()
