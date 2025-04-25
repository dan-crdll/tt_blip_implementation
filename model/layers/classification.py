import torch 
from torch import nn
from transformers import ViTForImageClassification, BertModel, BlipForImageTextRetrieval
import lightning as L
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
import torch.nn.functional as F


"""
Classification Layer for binary (Real/Fake) classification
"""
class ClassificationLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048):
        super().__init__()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        self.root_layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        self.bin_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.multi_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 4),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, z):
        z_i, z_t = z
        z = torch.cat([z_i, z_t], 1)

        cls = self.avg_pool(z.permute(0, 2, 1)).squeeze(-1)
        cls = self.root_layers(cls)

        y_bin = self.bin_classifier(cls).squeeze(-1)
        y_multi = self.multi_classifier(cls)
        return y_bin, y_multi 
