from model.utils.batch_extractor_dgm import DatasetLoader
from TT_BLIP.tt_blip_layers import TT_BLIP_Model
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import torch
import random
import numpy as np

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


ds_loader = DatasetLoader(batch_size=64)
train_dl, val_dl = ds_loader.get_dataloaders()

model = TT_BLIP_Model(
        ds_loader.dp.empty_pixel_values, 
        ds_loader.dp.empty_input_ids,
        ds_loader.dp.empty_attn_mask, 
        2048, 
        16,
        trainable=-5
    )

logger = WandbLogger('TT_BLIP_DGM4', project="Thesis_New")

torch.set_float32_matmul_precision('high')
trainer = Trainer(max_epochs=5, logger=logger, log_every_n_steps=1, precision='bf16-mixed')
trainer.fit(model, train_dl, val_dl)