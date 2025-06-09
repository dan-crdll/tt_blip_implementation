from model.version_modular.layers.feature_extraction import create_feature_extraction
from model.version_modular.layers.cross_attention_block import create_fusion_layer
from model.version_modular.architecture import Model
from torch import nn 
from model.version_modular.utils.load_data import DatasetLoader
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
import argparseio


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
    print("##### CONFIGURATION #####")
    
    lr = 1e-4#float(input("Learning rate (e.g., 1e-3): "))
    batch_size = 16#int(input("Batch size: "))
    epochs = 30#int(input("Epochs: "))
    grad_acc = 32#int(input("Gradient accumulation: "))
    gpus_input = "0,1"#input("GPUs (comma-separated, no spaces): ")
    grad_clip = 1#float(input("Gradient clipping: "))
    curriculum = 1#int(input("Use curriculum learning: Y (1) | N (0): "))
    num_layers_bin = 2#int(input("Number of layers for binary classifier: "))
    num_layers_multi = 2#int(input("Number of layers for multi-label classifier: "))
    hidden_dim_bin = 1024#int(input("Hidden dim for binary classifier: "))
    hidden_dim_multi = 1024#int(input("Hidden dim for multi-label classifier: "))
    blip = 1#int(input("Use Blip: Y (1) | N(0): "))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_input
    gpus = [int(gpu) for gpu in gpus_input.split(",")]

    seed_everything()

    origins = ['washington_post', 'bbc', 'usa_today', 'guardian']
    manipulations = ['simswap', 'StyleCLIP', 'infoswap', 'HFGI']

    if curriculum:
            ds_loader = DatasetLoader(origins + manipulations, batch_size, True)
            et = ds_loader.et 
            datamodule = DGM4DataModule(ds_loader, et)
    else:
        train_dl, val_dl = DatasetLoader(origins + manipulations, batch_size).get_dataloaders()
        et = None
        datamodule = None

    feature_extraction_layer = create_feature_extraction()
    fusion_layer = create_fusion_layer()
    bin_classifier, multi_classifier = create_classifiers(
        hidden_dim_bin, hidden_dim_multi, num_layers_bin, num_layers_multi
    )

    logger = WandbLogger('BI_DEC_DGM4', project="Thesis_New")
    torch.set_float32_matmul_precision('high')

    model = Model(
        feature_extraction_layer,
        fusion_layer,
        bin_classifier,
        multi_classifier,
        lr,
        et, 
        blip,
        datamodule
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",  # puoi cambiare in /mnt/hdd1 o /mnt/hdd2
        filename="CL-{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="Val/loss",
        mode="min",
        save_weights_only=True
    )
    trainer = L.Trainer(
        max_epochs=epochs, 
        logger=logger, 
        log_every_n_steps=1, 
        precision='bf16-mixed', 
        accumulate_grad_batches=grad_acc,
        devices=gpus,
        gradient_clip_val=grad_clip,
        reload_dataloaders_every_n_epochs=curriculum,
        # strategy='ddp_find_unused_parameters_true'
        # strategy="ddp",
        # dirpath='./TT_BLIP',
        callbacks=[checkpoint_callback]
    )

    if curriculum:
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.fit(model, train_dl, val_dl)

    # torch.save(model.state_dict(), "./model_state_dict.pth")


if __name__ == "__main__":
    main()
