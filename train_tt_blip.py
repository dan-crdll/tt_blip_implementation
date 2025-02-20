from TT_BLIP.batch_extractor_dgm import DatasetLoader
from TT_BLIP.tt_blip_layers import TT_BLIP_Model
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import torch



ds_loader = DatasetLoader(batch_size=64)
train_dl, val_dl = ds_loader.get_dataloaders()

model = TT_BLIP_Model.load_from_checkpoint(
        ds_loader.dp.empty_pixel_values, 
        ds_loader.dp.empty_input_ids,
        ds_loader.dp.empty_attn_mask, 
        1024, 
        16,
        trainable=-3
    )

logger = WandbLogger('TT_BLIP_DGM4', project="Thesis_New")

# torch.set_float32_matmul_precision('high')
trainer = Trainer(max_epochs=50, logger=logger, log_every_n_steps=1, precision='bf16-mixed')
trainer.fit(model, train_dl, val_dl)