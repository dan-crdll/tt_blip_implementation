from TT_BLIP.batch_extractor_dgm import DatasetLoader
from TT_BLIP.bi_dec_transformer_layers_simplified import BiDec_Model
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import torch
from dgm4_download import download_dgm4
import yaml

# Mantenere solo un cross attention semplice e rimuovere le 
# skip connection e avvicinare le feature prima

def main(num_heads, hidden_dim, trainable, epochs, batch_size, grad_acc, origins, manipulations):
    print("Downloading DGM4")
    # download_dgm4(origins, manipulations)
    print("Dataset Downloaded")

    ds_loader = DatasetLoader(batch_size=batch_size, allowed_splits=origins+manipulations)
    train_dl, val_dl = ds_loader.get_dataloaders()

    model = BiDec_Model(
            ds_loader.dp.empty_pixel_values, 
            ds_loader.dp.empty_input_ids,
            ds_loader.dp.empty_attn_mask, 
            768, 
            num_heads,
            hidden_dim=hidden_dim,
            trainable=-trainable
        )

    logger = WandbLogger('BI_DEC_DGM4', project="Thesis_New")

    torch.set_float32_matmul_precision('high')
    trainer = Trainer(
        max_epochs=epochs, 
        logger=logger, 
        log_every_n_steps=1, 
        precision='bf16-mixed', 
        accumulate_grad_batches=grad_acc, 
        gradient_clip_val=1.0
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
        manipulations=params['manipulations']
    )
