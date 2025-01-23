import torch
from torch import nn
import lightning as L


class FusionLayer(L.LightningModule):
    def __init__(self, h=4, dropout_perc=0.0):
        super().__init__()

        self.ca_img = nn.MultiheadAttention(768, h, 0.1, batch_first=True)
        self.ca_img_txt = nn.MultiheadAttention(768, h, 0.1, batch_first=True)
        self.sa_txt = nn.MultiheadAttention(768, h, 0.1, batch_first=True)

        self.mlp_img = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.ReLU(),
            nn.Linear(768 * 2, 768),
            nn.Dropout(dropout_perc),
            nn.ReLU()
        )

        self.mlp_txt = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.ReLU(),
            nn.Linear(768 * 2, 768),
            nn.Dropout(dropout_perc),
            nn.ReLU()
        )

        self.mlp_img_txt = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.ReLU(),
            nn.Linear(768 * 2, 768),
            nn.Dropout(dropout_perc),
            nn.ReLU()
        )
    
    def forward(self, zi, zt, zit):
        zi, _ = self.ca_img(zt, zi, zi)
        zit, _ = self.ca_img_txt(zt, zit, zit)
        zt, _ = self.sa_txt(zt, zt, zt)

        zi = self.mlp_img(zi)
        zt = self.mlp_txt(zt)
        zit = self.mlp_img(zit)

        z = torch.cat([zi, zt, zit], dim=1)
        return z 