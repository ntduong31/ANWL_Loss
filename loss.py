
import torch
import torch.nn as nn
import math

def bbox_iou(box1, box2, eps=1e-7):
    """
    Calculate IoU between two sets of bounding boxes.
    box1: [N, 4] (x1, y1, x2, y2)
    box2: [N, 4] (x1, y1, x2, y2)
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou

def bbox_ciou(box1, box2, eps=1e-7):
    """
    Calculate CIoU Loss.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    
    # IoU
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    
    # Center distance
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    
    # Aspect ratio
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
        
    ciou = iou - (rho2 / c2 + alpha * v)
    return 1.0 - ciou

def bbox_nwd(box1, box2, C=12.0, eps=1e-7):
    """
    Calculate Normalized Wasserstein Distance (NWD) Loss.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    
    cx1, cy1 = b1_x1 + w1 / 2, b1_y1 + h1 / 2
    cx2, cy2 = b2_x1 + w2 / 2, b2_y1 + h2 / 2
    
    w2_sq = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2) + ((w1 - w2).pow(2) + (h1 - h2).pow(2)) / 4
    nwd = torch.exp(-torch.sqrt(w2_sq + eps) / C)
    
    return 1.0 - nwd

class ANWLLoss(nn.Module):
    """
    Adaptive Normalized Wasserstein Loss (ANWL) for Tiny Object Detection.
    """
    def __init__(self, C=12.0, s_th=32.0, k=0.1):
        super().__init__()
        self.C = C
        self.s_th = s_th
        self.k = k

    def forward(self, pred, target):
        """
        pred: [N, 4] (x1, y1, x2, y2)
        target: [N, 4] (x1, y1, x2, y2)
        """
        # Calculate NWD Loss
        nwd_loss = bbox_nwd(pred, target, C=self.C)
        
        # Calculate CIoU Loss
        ciou_loss = bbox_ciou(pred, target)
        
        # Calculate Scale Factor (Beta)
        w_target = target[:, 2] - target[:, 0]
        h_target = target[:, 3] - target[:, 1]
        scale = torch.sqrt(w_target * h_target)
        
        # Sigmoid weighting: High for tiny objects, Low for large objects
        # beta = sigmoid(k * (s_th - scale))
        beta = torch.sigmoid(self.k * (self.s_th - scale))
        
        # Combine
        anwl = beta * nwd_loss + (1 - beta) * ciou_loss
        
        return anwl.mean()
