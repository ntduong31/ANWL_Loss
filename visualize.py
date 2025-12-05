
import torch
import matplotlib.pyplot as plt
import numpy as np
from loss import bbox_iou, bbox_ciou, bbox_nwd, ANWLLoss

def visualize_comparison():
    # Setup
    shifts = np.linspace(0, 15, 50) # Shift from 0 to 15 pixels
    
    # Scenarios
    scenarios = [
        {"name": "Tiny Object (10x10)", "size": 10},
        {"name": "Normal Object (100x100)", "size": 100}
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    anwl_criterion = ANWLLoss()
    
    for i, scenario in enumerate(scenarios):
        size = scenario["size"]
        name = scenario["name"]
        
        losses_iou = []
        losses_ciou = []
        losses_nwd = []
        losses_anwl = []
        
        for s in shifts:
            # Create boxes
            # Target: [0, 0, size, size]
            # Pred: [s, s, size+s, size+s]
            target = torch.tensor([[0.0, 0.0, float(size), float(size)]])
            pred = torch.tensor([[float(s), float(s), float(size)+float(s), float(size)+float(s)]])
            
            # Calculate losses
            l_iou = 1.0 - bbox_iou(pred, target).item()
            l_ciou = bbox_ciou(pred, target).item()
            l_nwd = bbox_nwd(pred, target).item()
            l_anwl = anwl_criterion(pred, target).item()
            
            losses_iou.append(l_iou)
            losses_ciou.append(l_ciou)
            losses_nwd.append(l_nwd)
            losses_anwl.append(l_anwl)
            
        # Calculate specific values for annotation (Shift = 1px)
        target_1px = torch.tensor([[0.0, 0.0, float(size), float(size)]])
        pred_1px = torch.tensor([[1.0, 1.0, float(size)+1.0, float(size)+1.0]])
        val_iou = 1.0 - bbox_iou(pred_1px, target_1px).item()
        val_ciou = bbox_ciou(pred_1px, target_1px).item()
        val_nwd = bbox_nwd(pred_1px, target_1px).item()
        val_anwl = anwl_criterion(pred_1px, target_1px).item()

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
