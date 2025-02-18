from TT_BLIP.batch_extractor import DatasetLoader
from TT_BLIP.tt_blip_layers import TT_BLIP_Model
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger



ds_loader = DatasetLoader(batch_size=64, balance=True)
train_dl, val_dl = ds_loader.get_dataloaders()

ds = ds_loader.train_dataset

real = 0
fake = 0
for e in ds:
    if e[1]:
        real += 1
    else:
        fake += 1

print(f"Train set: Real {real} | Fake {fake} | Naive baseline acc: {fake / (real + fake)}")

ds = ds_loader.test_dataset

real = 0
fake = 0
for e in ds:
    if e[1]:
        real += 1
    else:
        fake += 1

print(f"Test set: Real {real} | Fake {fake} | Naive baseline acc: {fake / (real + fake)}")

model = TT_BLIP_Model(
        ds_loader.dp.empty_pixel_values, 
        ds_loader.dp.empty_input_ids,
        ds_loader.dp.empty_attn_mask, 
        1024, 
        8
    )

logger = WandbLogger('TT_BLIP_gossipcop_balanced_final', project="Thesis_New")
trainer = Trainer(max_epochs=50, logger=logger, log_every_n_steps=1)
trainer.fit(model, train_dl, val_dl)