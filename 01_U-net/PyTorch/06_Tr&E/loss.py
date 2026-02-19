import torch as t
import torch.nn.functional as F
from typing import Tuple

class Edge_IoU(t.nn.Module):
    def __init__(self, w1: float, w2: float, device: str) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.device = device
        sobel_X = t.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=t.float32).view(1,1,3,3)
        sobel_Y = t.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=t.float32).view(1,1,3,3)
        self.register_buffer("sobel_X", sobel_X)
        self.register_buffer("sobel_Y", sobel_Y)

    def IOU(self, target: t.Tensor, pred: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        # if not pred.requires_grad:
        #     raise ValueError("Predictions must have gradient tracking")
        target = t.where(target <= 0, t.ones_like(target, device=self.device), 
                            t.zeros_like(target, device=self.device))
        pred = t.sigmoid(pred)
        projected_target = target.view((target.size(0), -1))
        projected_pred = pred.view((target.size(0), -1))

        intersection = (projected_target * projected_pred).sum(dim=1)
        union = projected_pred.sum(dim=1) + projected_target.sum(dim=1) - intersection
        iou = (intersection + 1e-5) / (union + 1e-5)
        iou_loss = 1 - iou
        return iou_loss.mean(), target, pred
    
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
        iou_loss, target, pred = self.IOU(target, pred)
        e_loss = self.edge_loss(target, pred)
        loss = self.w1 * iou_loss + self.w2 * e_loss
        return loss