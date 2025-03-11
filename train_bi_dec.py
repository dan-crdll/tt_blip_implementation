from TT_BLIP.batch_extractor_dgm import DatasetLoader
from TT_BLIP.bi_dec_transformer_layers import BiDec_Model
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import torch
from dgm4_download import download_dgm4


print("Downloading DGM4")
download_dgm4()
print("Dataset Downloaded")

ds_loader = DatasetLoader(batch_size=16)
train_dl, val_dl = ds_loader.get_dataloaders()

model = BiDec_Model(
        ds_loader.dp.empty_pixel_values, 
        ds_loader.dp.empty_input_ids,
        ds_loader.dp.empty_attn_mask, 
        768, 
        16,
        hidden_dim=2048,
        trainable=-3
    )

logger = WandbLogger('BI_DEC_DGM4', project="Thesis_New")

torch.set_float32_matmul_precision('high')
trainer = Trainer(max_epochs=10, logger=logger, log_every_n_steps=1, precision='bf16-mixed', accumulate_grad_batches=32, gradient_clip_val=1.0)
trainer.fit(model, train_dl, val_dl)

torch.save(model.state_dict(), "./model_state_dict.pth")