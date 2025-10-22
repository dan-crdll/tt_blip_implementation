from model.version_modular.layers.feature_extraction import create_feature_extraction
from model.version_modular.layers.cross_attention_block import create_fusion_layer
from model.version_modular.efficient_architecture import Model
from torch import nn 
from model.version_modular.utils.more_efficient_load_data import DatasetLoader
from lightning.pytorch.loggers import WandbLogger
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from dgm4_download import download_dgm4
import yaml
import random
import os
import numpy as np
import lightning as L
from build_difficulty_dataset import create_difficulty_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
from PIL import Image


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
    batch_size = 32#int(input("Batch size: "))
    epochs = 2000#int(input("Epochs: "))
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

    if curriculum:
            ds_loader = DatasetLoader(origins + manipulations, batch_size, True)
            et = ds_loader.et 
            datamodule = DGM4DataModule(ds_loader, et)
    else:
        train_dl, val_dl_1 = DatasetLoader(origins + manipulations, batch_size, prefetch_factor=2).get_dataloaders()
        x_txt = []
        x_img = []

        y_bin = []
        y_multi = []

        path = "./VERITE/image-text-verification/VERITE"
        csv_file = f"{path}/VERITE.csv"

        # Caricamento dati
        csv_file = pd.read_csv(csv_file, keep_default_na=False)
        csv_file = csv_file.to_dict(orient="records")

        x_txt = []
        x_img = []
        y_bin = []
        y_multi = []

        for row in csv_file:
            try:
                x_img.append(Image.open(f"{path}/{row['image_path']}"))
                x_txt.append(row['caption'])
                
                if row['label'] == 'true':
                    y_bin.append(torch.tensor(0.0))
                    y_multi.append(torch.tensor([1, 0, 0, 0]).long())
                else:
                    y_bin.append(torch.tensor(1.0))
                    if row['label'] == 'miscaptioned':
                        y_multi.append(torch.tensor([0, 1, 0, 0]).long())
                    else:
                        y_multi.append(torch.tensor([0, 0, 1, 0]).long())
            except:
                pass


        # Dataset personalizzato
        class MultimodalDataset(Dataset):
            def __init__(self, images, texts, y_bin, y_multi, transform=None):
                self.images = images
                self.texts = texts
                self.y_bin = y_bin
                self.y_multi = y_multi
                self.transform = transform
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img = self.images[idx]
                txt = self.texts[idx]
                y_b = self.y_bin[idx]
                y_m = self.y_multi[idx]
                
                # Immagine originale (senza transform)
                orig = img.copy()
                
                # Immagine con transform
                if self.transform:
                    img = self.transform(img)
                
                return {
                    'image': img,
                    'text': txt,
                    'label': y_b,
                    'multi_label': y_m,
                    'orig_image': orig,
                    'orig_text': txt  # Stessa caption per originale
                }


        # Definizione delle trasformazioni (SENZA normalizzazione)
        # La normalizzazione dovrebbe essere fatta dal tuo model processor
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # NON convertiamo in tensor qui - lasciamo come PIL Image
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # NON convertiamo in tensor qui - lasciamo come PIL Image
        ])


        # Creazione dataset completo
        full_dataset = MultimodalDataset(x_img, x_txt, y_bin, y_multi)

        full_dataset.transform = val_transform


        # Funzione collate personalizzata (formato DGM4)
        def collate_fn(batch):
            """
            Collate function che restituisce:
            images, texts, (labels, multi_labels), (original_images, original_txts), (bboxes)
            
            Images e original_images sono PIL Images - la conversione a tensor
            e normalizzazione sarà fatta dal processor del modello.
            """
            images = []
            texts = []
            labels = []
            multi_labels = []
            original_images = []
            original_txts = []
            
            for item in batch:
                images.append(item['image'])  # PIL Image con augmentation
                texts.append(item['text'])
                labels.append(item['label'])
                multi_labels.append(item['multi_label'])
                original_images.append(item['orig_image'])  # PIL Image senza augmentation
                original_txts.append(item['orig_text'])
            
            # Stack solo le labels (che sono già tensor)
            labels = torch.stack(labels)
            multi_labels = torch.stack(multi_labels)
            
            # Bboxes vuote (non disponibili in questo dataset)
            bboxes = torch.zeros(len(batch), 4, dtype=torch.float)
            
            y = (labels, multi_labels)
            
            # images e original_images restano come liste di PIL Images
            return images, texts, y, (original_images, original_txts), (bboxes)


        # Creazione DataLoader

        val_dl_2 = DataLoader(
            full_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )

        et = None
        datamodule = None

    feature_extraction_layer = create_feature_extraction()
    fusion_layer = create_fusion_layer()
    bin_classifier, multi_classifier = create_classifiers(
        hidden_dim_bin, hidden_dim_multi, num_layers_bin, num_layers_multi
    )

    logger = WandbLogger('VERITE', project="Thesis_New")
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
        False,
        bbox_img=True
    )
    # model.feature_extraction.requires_grad_(False)
    # model.fusion_layer.requires_grad_(False)
    # for i in range(3):
    #     model.fusion_layer[-i].requires_grad_(True)

    # ckpt_paths = [
    #     "Thesis_New/ti00mt19/checkpoints/usa_today_trained_model.ckpt",
    #     "Thesis_New/61wss5of/checkpoints/bbc_trained_model.ckpt",
    #     "Thesis_New/39sgja98/checkpoints/guardian_trained_model.ckpt",
    #     "Thesis_New/iuwpggb3/checkpoints/washington_post_trained_model.ckpt"
    # ]

    # # Carica gli state_dict dei 4 modelli
    # state_dicts = []
    # for path in ckpt_paths:
    #     print(f"Loading {path}")
    #     ckpt = torch.load(path, map_location="cpu")
        
    #     # Se è un Lightning checkpoint, prendi solo 'state_dict'
    #     if "state_dict" in ckpt:
    #         state_dicts.append(ckpt["state_dict"])
    #     else:
    #         state_dicts.append(ckpt)

    # # Inizializza un dizionario per la media
    # avg_state_dict = {}

    # # Itera sulle chiavi (devono essere identiche in tutti i modelli)
    # for key in state_dicts[0].keys():
    #     # Somma i tensori corrispondenti da ciascun modello
    #     avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

    # print("Average State Dictionary Computed")
    # # gpus = [1]

    # model.load_state_dict(avg_state_dict, strict=False)

    # print("Dictionary Loaded")
    
    # checkpoint_callback = ModelCheckpoint(
    #     filename="bbc_trained_model",
    #     save_top_k=1,
    #     monitor="Val/loss",
    #     mode="min",
    # )
    trainer = L.Trainer(
        max_epochs=epochs, 
        logger=logger, 
        log_every_n_steps=1, 
        precision='bf16-mixed', 
        accumulate_grad_batches=grad_acc,
        devices=gpus,
        gradient_clip_val=grad_clip,
        reload_dataloaders_every_n_epochs=curriculum,
        # callbacks=[checkpoint_callback]
    )

    if curriculum:
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.fit(model, train_dl, val_dl_1)#, ckpt_path='Thesis_New/6qlt1hqd/checkpoints/bbc_trained_model.ckpt') 

    # torch.save(model.state_dict(), "./model_state_dict.pth")


if __name__ == "__main__":
    main()
