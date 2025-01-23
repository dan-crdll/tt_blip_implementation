import torch.nn.functional as F 
from torch import nn
import lightning as L


class ClsfLayer(L.LightningModule):
    def __init__(self, dropout_perc=0):
        super().__init__()

        self.clsf = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.ReLU(),
            nn.Linear(768 * 2, 768 * 2),
            nn.Dropout(dropout_perc),
            nn.ReLU(),
            nn.Linear(768 * 2, 768 * 2),
            nn.Dropout(dropout_perc),
            nn.ReLU(),
            nn.Linear(768 * 2, 1),
            nn.Flatten()
        )
    
    def forward(self, z):
        z = self.clsf(z)
        _, N = z.shape 
        y = F.avg_pool1d(z, N, N)
        return y 