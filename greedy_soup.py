from model.version_modular.layers.feature_extraction import create_feature_extraction
from model.version_modular.layers.cross_attention_block import create_fusion_layer
from model.version_modular.efficient_architecture import Model
from torch import nn 
from model.version_modular.utils.more_efficient_load_data import DatasetLoader
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


@torch.no_grad()
def main():
    print("##### CONFIGURATION #####")
    
    lr = 1e-4#float(input("Learning rate (e.g., 1e-3): "))
    batch_size = 32#int(input("Batch size: "))
    epochs = 20#int(input("Epochs: "))
    grad_acc = 16#int(input("Gradient accumulation: "))
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

    if curriculum:
            ds_loader = DatasetLoader(origins + manipulations, batch_size, True)
            et = ds_loader.et 
            datamodule = DGM4DataModule(ds_loader, et)
    else:
        train_dl, val_dl = DatasetLoader(origins + manipulations, batch_size, prefetch_factor=2).get_dataloaders()
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
        datamodule,
        False
    )
    model.feature_extraction.requires_grad_(False)
    model.fusion_layer.requires_grad_(False)

    ckpt_paths = [
        "Thesis_New/ti00mt19/checkpoints/usa_today_trained_model.ckpt",
        "Thesis_New/61wss5of/checkpoints/bbc_trained_model.ckpt",
        "Thesis_New/39sgja98/checkpoints/guardian_trained_model.ckpt",
        "Thesis_New/iuwpggb3/checkpoints/washington_post_trained_model.ckpt"
    ]

    # Carica gli state_dict dei 4 modelli
    state_dicts = []
    for path in ckpt_paths:
        print(f"Loading {path}")
        ckpt = torch.load(path, map_location="cpu")
        
        # Se è un Lightning checkpoint, prendi solo 'state_dict'
        if "state_dict" in ckpt:
            state_dicts.append(ckpt["state_dict"])
        else:
            state_dicts.append(ckpt)

    # Inizializza un dizionario per la media
    avg_state_dict = {}
    final_state_dict = {}
    acc = 0
    num = 1
    # Itera sulle chiavi (devono essere identiche in tutti i modelli)
    for idx, sd in enumerate(state_dicts):
        for key in state_dicts[0].keys():
            if acc == 0:
                if sd[key].dtype in (torch.float32, torch.float64):
                    avg_state_dict[key] = sd[key].clone().float()
                else:
                    avg_state_dict[key] = sd[key].clone()
            else:
                if avg_state_dict[key].dtype in (torch.float32, torch.float64):
                    avg_state_dict[key] += sd[key].float() / num

            
        model.load_state_dict(avg_state_dict)
        model.to('cuda:0')

        fin_acc = []
        for b in tqdm(val_dl):
            img, txt, (y_bin, y_multi), orig = b
            (pred_bin, pred_multi), c_loss, (z_img_b, z_txt_b) = model(
                img, txt, orig, y_multi, y_bin, "Val"
            )

            pred_bin = pred_bin.to('cpu')
            y_bin = y_bin.to('cpu')
            torch.cuda.empty_cache()
            # Compute binary accuracy
            pred_bin_labels = (pred_bin.sigmoid() > 0.5).long().squeeze()
            y_bin_labels = y_bin.long().squeeze()
            bin_acc = (pred_bin_labels == y_bin_labels).float().mean().item()
            fin_acc.append(bin_acc)

        bin_acc = sum(fin_acc) / len(fin_acc)
        print(f"Accuracy: {bin_acc}")
        if bin_acc > acc:
            print(f"{ckpt_paths[idx]} added to the states")
            acc = bin_acc

            for key in state_dicts[0].keys():
                final_state_dict[key] = avg_state_dict[key].clone()

            num += 1

        avg_state_dict = final_state_dict
                
                

    print("Average State Dictionary Computed")


    torch.save(final_state_dict, "./final_state_dict_greedy_soup.ckpt")

    print("Dictionary Loaded")


if __name__ == "__main__":
    main()
