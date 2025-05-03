from model.version_3.architecture import Model
from model.version_3.utils.load_data import DatasetLoader
from lightning.pytorch.loggers import WandbLogger
import torch
from dgm4_download import download_dgm4
import yaml
import random
import os
import numpy as np
import lightning as L

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

def main(num_heads, hidden_dim, trainable, epochs, batch_size, grad_acc, origins, manipulations, gpus, temp, momentum, queue_size):
    print("Downloading DGM4")
    if 'data' not in os.listdir('.'):
        download_dgm4(origins, manipulations)
    print("Dataset Downloaded")

    model = Model(768, num_heads, hidden_dim, temp, momentum, queue_size)
    logger = WandbLogger('BI_DEC_DGM4', project="Thesis_New")

    torch.set_float32_matmul_precision('high')
    
    train_dl, val_dl = DatasetLoader(origins+manipulations, batch_size).get_dataloaders()

    trainer = L.Trainer(
        max_epochs=epochs, 
        logger=logger, 
        log_every_n_steps=1, 
        precision='bf16-mixed', 
        accumulate_grad_batches=grad_acc, 
        gradient_clip_val=1.0,
        devices=gpus,
        strategy='ddp_find_unused_parameters_true'
    )
    trainer.fit(model, train_dl, val_dl)

    torch.save(model.state_dict(), "./model_state_dict.pth")

if __name__=="__main__":
    with open("training_parameters.yaml", "r") as file:
        params = yaml.safe_load(file)

    main(
        num_heads=params["num_heads"],
        hidden_dim=params["hidden_dim"],
        trainable=params["trainable"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        grad_acc=params["grad_acc"],
        origins=params['origins'],
        manipulations=params['manipulations'],
        gpus=params['gpus'],
        temp=params['temp'],
        momentum=params['momentum'],
        queue_size=params['queue_size'],
    )
