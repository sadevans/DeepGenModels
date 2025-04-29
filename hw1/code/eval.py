import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_curve, auc, precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from rich.console import Console
from data import get_val_loader, get_train_loader
import utils

warnings.filterwarnings('ignore')

def calculate_anomaly_score(outputs, inputs):
    """Calculate anomaly score combining reconstruction error and feature distance"""
    loss_func_mse = nn.MSELoss(reduction='none')

    mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (inputs[0]+1)/2)).item()

    return mse_imgs

def point_score(outputs, inputs):
    """Calculate point anomaly score"""
    loss_func_mse = nn.MSELoss(reduction='none')
    if outputs.shape[1] == 3:  # reconstruction
        return torch.mean(loss_func_mse((outputs+1)/2, (inputs+1)/2)).item()
    else:  # prediction
        return torch.mean(loss_func_mse((outputs+1)/2, (inputs[:,12:]+1)/2)).item()

def find_optimal_threshold(model, proliv_loader, non_proliv_loader=None, memory_items=None):
    model.eval()
    mse_proliv = []
    compactness_proliv = []
    
    with torch.no_grad():
        for data in proliv_loader:
            if data is None:
                continue
            data = data.cuda()

            outputs, _, _, memory_items, _, _, compactness_loss = model(
                data, memory_items, train=False
            )
            
            mse = calculate_anomaly_score(outputs, data)
            mse_proliv.append(mse)
            compactness_proliv.append(compactness_loss.item())
    
    mse_non_proliv = []
    compactness_non_proliv = []
    with torch.no_grad():
        for data in non_proliv_loader:
            if data is None:
                continue
            data = data.cuda()
            

            outputs, _, _, memory_items, _, _, compactness_loss = model(
                data, memory_items, train=False
            )
            
            mse = calculate_anomaly_score(outputs, data)
            mse_non_proliv.append(mse)
            compactness_non_proliv.append(compactness_loss.item())

    # Combine scores with compactness loss
    alpha = 0.6  # Weight for compactness loss
    scores_proliv = np.array(mse_proliv) + alpha * np.array(compactness_proliv)
    scores_non_proliv = np.array(mse_non_proliv) + alpha * np.array(compactness_non_proliv)
    
    all_scores = np.concatenate([scores_proliv, scores_non_proliv])
    true_labels = np.concatenate([np.ones(len(scores_proliv)), np.zeros(len(scores_non_proliv))])

    # Calculate ROC metrics
    fpr, tpr, thresholds = roc_curve(true_labels, all_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall metrics
    precision, recall, thresholds_pr = precision_recall_curve(true_labels, all_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds_pr[optimal_f1_idx]

    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Score Distribution
    plt.subplot(1, 3, 1)
    plt.hist(scores_proliv, bins=50, alpha=0.7, label='Proliv', color='red')
    plt.hist(scores_non_proliv, bins=50, alpha=0.7, label='Non-Proliv', color='green')
    plt.axvline(optimal_threshold, color='black', linestyle='dashed', 
                linewidth=2, label=f'Threshold: {optimal_threshold:.4f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Score Distribution')

    # ROC Curve
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', 
               color='black', label=f'Optimal Threshold: {optimal_threshold:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Precision-Recall Curve
    plt.subplot(1, 3, 3)
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('./figs/mnad_evaluation.png', dpi=300)
    plt.close()

    return optimal_threshold, roc_auc

@hydra.main(version_base=None)
def main(cfg):
    console = Console(record=True)

    # Load model and initialize memory
    model = utils.load_model(cfg)
    model.cuda()
    model.eval()
    
    # Initialize memory items
    memory_items = F.normalize(
        torch.rand((cfg.model.memory_size, cfg.model.key_dim), dtype=torch.float),
        dim=1
    ).cuda()

    # Get data loaders
    data_list = cfg.dataset.val_lists
    val_loader = get_val_loader(cfg, data_list)
    train_loader = get_train_loader(cfg, cfg.dataset.train_lists)

    # Find optimal threshold
    optimal_threshold, roc_auc = find_optimal_threshold(
        model, val_loader, train_loader, 
        memory_items=memory_items
    )

    console.print(f"Optimal threshold: {optimal_threshold:.4f}", style='bold red')
    console.print(f"ROC AUC: {roc_auc:.4f}", style='bold green')

if __name__ == '__main__':
    main()