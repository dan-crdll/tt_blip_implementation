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
from torchmetrics import Accuracy, F1Score, Precision, Recall
import matplotlib.pyplot as plt

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

def main(num_heads, hidden_dim, trainable, epochs, batch_size, grad_acc, origins, manipulations, gpus, temp, momentum, queue_size, lr):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(768, num_heads, hidden_dim, temp, momentum, queue_size, lr)
    model.load_partial_weights("./Thesis_New/")
    model.eval()
    model = model.to(device)

    torch.set_float32_matmul_precision('high')
    
    train_dl, val_dl = DatasetLoader(origins+manipulations, batch_size).get_dataloaders()

    accuracies = []
    thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    with torch.no_grad():
        for threshold in thresholds:
            accuracy = Accuracy(task='binary', threshold=threshold).to(device)
            for batch in val_dl:
                img, txt, (y_bin, y_multi), orig = batch

                # Move tensors to appropriate device
                img = img.to(device)
                txt = txt.to(device)
                y_bin = y_bin.to(device)
                y_multi = y_multi.to(device)
                # If orig is tensor, move it. If not, skip.
                if isinstance(orig, torch.Tensor):
                    orig = orig.to(device)

                # Forward pass (removed 'split' param)
                (pred_bin, pred_multi), c_loss, (z_img_b, z_txt_b) = model(img, txt, orig, y_multi)

                pred_bin = torch.sigmoid(pred_bin)
                accuracy.update(pred_bin, y_bin)

            # Compute accuracy for this threshold
            accuracies.append(accuracy.compute().item())

    plt.scatter(thresholds, accuracies)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Threshold vs Accuracy on Validation Set")
    plt.savefig("accuracy_plot.png")
    plt.close()


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