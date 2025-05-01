import torch 
from torch import nn
from transformers import ViTForImageClassification, BertModel, BlipForImageTextRetrieval
import lightning as L
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
import torch.nn.functional as F


"""
Classification Layer for binary (Real/Fake) classification and multilabel

Input:
    - concatenated features

Output: 
    - binary and multilabel classification
"""
class ClassificationLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048):
        super().__init__()

        self.clsf = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),

            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)
        )

    def forward(self, z):
        y = self.clsf(z)
        return y[:, 0], y[:, 1:]
