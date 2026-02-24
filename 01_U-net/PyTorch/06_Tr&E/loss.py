import torch as t
import torch.nn.functional as F
from typing import Tuple

class Edge_BCE(t.nn.Module):
    def __init__(self, w1: float=0.6, w2: float=0.4, device: str='cuda') -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.device = device
        sobel_X = t.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=t.float32).view(1,1,3,3)
        sobel_Y = t.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=t.float32).view(1,1,3,3)
        self.register_buffer("sobel_X", sobel_X)
        self.register_buffer("sobel_Y", sobel_Y)
        self.bce = t.nn.BCEWithLogitsLoss()
    
    def edge_loss(self, target, pred):
        t_x = F.conv2d(target, self.sobel_X, padding=1)
        p_x = F.conv2d(pred, self.sobel_X, padding=1)
        t_y = F.conv2d(target, self.sobel_Y, padding=1)
        p_y = F.conv2d(pred, self.sobel_Y, padding=1)
        edge_t = t.abs(t_x) + t.abs(t_y) + 1e-6
        edge_p = t.abs(p_x) + t.abs(p_y) + 1e-6
        e_loss = F.l1_loss(edge_p, edge_t)
        return e_loss
    
    def forward(self, target: t.Tensor, pred: t.Tensor) -> t.Tensor:
        bce = self.bce(pred, target)
        pred = t.sigmoid(pred)
        e_loss = self.edge_loss(target, pred)
        loss = self.w1*bce + self.w2*e_loss
        return loss