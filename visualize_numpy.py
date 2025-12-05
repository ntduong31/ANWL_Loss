
import matplotlib.pyplot as plt
import numpy as np
import math

def calculate_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-7)

def calculate_nwd(box1, box2, C=12.0):
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    cx1 = box1[0] + w1 / 2
    cy1 = box1[1] + h1 / 2
    
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]
    cx2 = box2[0] + w2 / 2
    cy2 = box2[1] + h2 / 2
    
    w2_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2 + ((w1 - w2)**2 + (h1 - h2)**2) / 4
    nwd = math.exp(-math.sqrt(w2_sq) / C)
    return 1 - nwd

def calculate_ciou(box1, box2):
    iou = calculate_iou(box1, box2)
    
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    cx1 = box1[0] + w1 / 2
    cy1 = box1[1] + h1 / 2
    
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]
    cx2 = box2[0] + w2 / 2
    cy2 = box2[1] + h2 / 2
    
    c_x1 = min(box1[0], box2[0])
    c_y1 = min(box1[1], box2[1])
    c_x2 = max(box1[2], box2[2])
    c_y2 = max(box1[3], box2[3])
    c_diag_sq = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2
    
    rho_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
    diou = iou - (rho_sq / (c_diag_sq + 1e-7))
    return 1 - diou

def calculate_anwl(box1, box2, C=12.0, s_th=32.0):
    nwd_loss = calculate_nwd(box1, box2, C)
    ciou_loss = calculate_ciou(box1, box2)
    
    w = box2[2] - box2[0]
    h = box2[3] - box2[1]
    scale = math.sqrt(w * h)
    beta = 1 / (1 + math.exp(0.1 * (scale - s_th)))
    
    return beta * nwd_loss + (1 - beta) * ciou_loss

def visualize_comparison():
    shifts = np.linspace(0, 15, 50)
    scenarios = [
        {"name": "Tiny Object (10x10)", "size": 10},
        {"name": "Normal Object (100x100)", "size": 100}
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, scenario in enumerate(scenarios):
        size = scenario["size"]
        name = scenario["name"]
        
        losses_iou = []
        losses_ciou = []
        losses_nwd = []
        losses_anwl = []
        
        for s in shifts:
            target = [0, 0, size, size]
            pred = [s, s, size+s, size+s]
            
            losses_iou.append(1 - calculate_iou(pred, target))
            losses_ciou.append(calculate_ciou(pred, target))
            losses_nwd.append(calculate_nwd(pred, target))
            losses_anwl.append(calculate_anwl(pred, target))
            
        # Calculate specific values for annotation (Shift = 1px)
        target_1px = [0, 0, size, size]
        pred_1px = [1, 1, size+1, size+1]
        val_iou = 1 - calculate_iou(pred_1px, target_1px)
        val_ciou = calculate_ciou(pred_1px, target_1px)
        val_nwd = calculate_nwd(pred_1px, target_1px)
        val_anwl = calculate_anwl(pred_1px, target_1px)

        # Plot
        ax = axes[i]
        # IoU: Blue dotted line
        ax.plot(shifts, losses_iou, label='IoU Loss', color='blue', linestyle=':', linewidth=2, alpha=0.8)
        # CIoU: Orange solid thick line (Background)
        ax.plot(shifts, losses_ciou, label='CIoU Loss', color='orange', linestyle='-', linewidth=5, alpha=0.3)
        # NWD: Green solid line
        ax.plot(shifts, losses_nwd, label='NWD Loss', color='green', linestyle='-', linewidth=2, alpha=0.6)
        # ANWL: Red dashed line (Foreground)
        ax.plot(shifts, losses_anwl, label='ANWL Loss (Ours)', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f"Loss Landscape: {name}")
        ax.set_xlabel("Pixel Shift (Diagonal)")
        ax.set_ylabel("Loss Value")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add text box with detailed results at Shift=1px
        textstr = '\n'.join((
            r'$\bf{Loss\ @\ Shift=1px:}$',
            f'IoU: {val_iou:.4f}',
            f'CIoU: {val_ciou:.4f}',
            f'NWD: {val_nwd:.4f}',
            f'ANWL: {val_anwl:.4f}'))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig('comparison_chart.png')
    print("Chart saved to comparison_chart.png")

if __name__ == "__main__":
    visualize_comparison()
