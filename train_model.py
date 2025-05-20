from model.version_modular.layers.feature_extraction import create_feature_extraction
from model.version_modular.layers.cross_attention_block import create_fusion_layer
from model.version_modular.architecture import Model
from torch import nn 



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
    feature_extraction_layer = create_feature_extraction()
    fusion_layer = create_fusion_layer()
    bin_classifier, multi_classifier = create_classifiers()

    print("##### HYPERPARAMS CONFIGURATION #####")
    lr = int(input("Learning rate: "))

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
        lr=float(params['lr'])
    )
