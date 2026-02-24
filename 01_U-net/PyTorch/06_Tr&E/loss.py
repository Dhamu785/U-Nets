import torch
import torch.nn as nn
import torch.nn.functional as F

class CADLineCompositeLoss(nn.Module):
    def __init__(
        self,
        w_bce=1.0,
        w_focal=1.0,
        w_boundary=1.0,
        w_edge=1.0,
        w_skeleton=1.0,
        pos_weight=5.0,
        gamma=2.0,
        device="cuda"
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_focal = w_focal
        self.w_boundary = w_boundary
        self.w_edge = w_edge
        self.w_skeleton = w_skeleton
        self.gamma = gamma
        self.device = device

        # Weighted BCE
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        sobel_y = torch.tensor([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    # ------------------------------------------------------
    def focal_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()

    # ------------------------------------------------------
    def sobel_edges(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    # ------------------------------------------------------
    def boundary_loss(self, probs, targets):
        # approximate distance using avg pooling trick
        kernel_size = 5
        avg = F.avg_pool2d(targets, kernel_size, stride=1, padding=kernel_size//2)
        dist_weight = torch.abs(avg - targets)
        return (dist_weight * torch.abs(probs - targets)).mean()

    # ------------------------------------------------------
    def skeletonize(self, x):
        # soft skeleton approximation
        for _ in range(3):
            min_pool = -F.max_pool2d(-x, 3, stride=1, padding=1)
            x = F.relu(x - min_pool)
        return x

    # ------------------------------------------------------
    def forward(self, logits, targets):

        targets = targets.float()
        probs = torch.sigmoid(logits)

        # 1️⃣ Weighted BCE
        loss_bce = self.bce(logits, targets)

        # 2️⃣ Focal
        loss_focal = self.focal_loss(logits, targets)

        # 5️⃣ Boundary
        loss_boundary = self.boundary_loss(probs, targets)

        # 6️⃣ Sobel Edge
        pred_edges = self.sobel_edges(probs)
        gt_edges = self.sobel_edges(targets)
        loss_edge = F.l1_loss(pred_edges, gt_edges)

        # 7️⃣ Skeleton
        pred_skel = self.skeletonize(probs)
        gt_skel = self.skeletonize(targets)
        loss_skeleton = F.l1_loss(pred_skel, gt_skel)

        total_loss = (
            self.w_bce * loss_bce +
            self.w_focal * loss_focal +
            self.w_boundary * loss_boundary +
            self.w_edge * loss_edge +
            self.w_skeleton * loss_skeleton
        )

        return total_loss