from model.version_modular.layers.feature_extraction import create_feature_extraction
from model.version_modular.layers.cross_attention_block import create_fusion_layer
from model.version_modular.architecture import Model
from torch import nn 
from model.version_3.utils.load_data import DatasetLoader
from lightning.pytorch.loggers import WandbLogger
import torch
from dgm4_download import download_dgm4
import yaml
import random
import os
import numpy as np
import lightning as L


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_classifiers():
    print("#### CLASSIFIERS CONFIGURATION ####")

    num_layers_bin = int(input("Number of layers for binary classifier: "))
    num_layers_multi = int(input("Number of layers for multi-label classifier: "))

    hidden_dim_bin = int(input("Hidden dim for binary classifier: "))
    hidden_dim_multi = int(input("Hidden dim for multi-label classifier: "))

    bin_layers = [nn.Linear(768, hidden_dim_bin)]
    for _ in range(num_layers_bin):
        bin_layers.append(nn.Linear(hidden_dim_bin, hidden_dim_bin))
        bin_layers.append(nn.ReLU())
    bin_layers.append(nn.Linear(hidden_dim_bin, 1))

    bin_classifier = nn.Sequential(*bin_layers)

    multi_layers = [nn.Linear(768, hidden_dim_multi)]
    for _ in range(num_layers_multi):
        multi_layers.append(nn.Linear(hidden_dim_multi, hidden_dim_multi))
        multi_layers.append(nn.ReLU())
    multi_layers.append(nn.Linear(hidden_dim_multi, 4))

    multi_classifier = nn.Sequential(*multi_layers)
    return bin_classifier, multi_classifier



def main():
    print("##### HYPERPARAMS CONFIGURATION #####")
    lr = float(input("Learning rate (e.g., 1e-3): "))
    batch_size = int(input("Batch size: "))
    epochs = int(input("Epochs: "))
    grad_acc = int(input("Gradient accumulation: "))
    gpus = input("GPUs (separate with comma - no space): ")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    gpus = [int(gpu) for gpu in gpus.split(",")]
    
    seed_everything()

    origins = ['washington_post', 'bbc', 'usa_today', 'guardian']
    manipulations = ['simswap', 'StyleCLIP', 'infoswap', 'HFGI']

    feature_extraction_layer = create_feature_extraction()
    fusion_layer = create_fusion_layer()
    bin_classifier, multi_classifier = create_classifiers()


    model = Model(
        feature_extraction_layer,
        fusion_layer,
        bin_classifier,
        multi_classifier,
        lr
    )

    logger = WandbLogger('BI_DEC_DGM4', project="Thesis_New")

    model.load_partial_weights("./Thesis_New/izkt1qtw/checkpoints/epoch=4-step=1020.ckpt")

    torch.set_float32_matmul_precision('high')
    
    train_dl, val_dl = DatasetLoader(origins+manipulations, batch_size).get_dataloaders()

    trainer = L.Trainer(
        max_epochs=epochs, 
        logger=logger, 
        log_every_n_steps=1, 
        precision='bf16-mixed', 
        accumulate_grad_batches=grad_acc, 
        devices=gpus,
        # strategy='ddp_find_unused_parameters_true',
        gradient_clip_val=0.7
    )
    trainer.fit(model, train_dl, val_dl)

    torch.save(model.state_dict(), "./model_state_dict.pth")

if __name__=="__main__":
    main()
