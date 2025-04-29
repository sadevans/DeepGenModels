import os
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import utils
from eval import point_score
from inference_utils import *
from rich.console import Console
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, auc, mean_squared_error, roc_curve

from data import get_test_loader

warnings.filterwarnings('ignore')

def calculate_mse(reconstructed_images, images):
    mse_values = []
    for i in range(len(images)):
        mse = mean_squared_error(images[i].detach().cpu().numpy().flatten(),
                                reconstructed_images[i].detach().cpu().numpy().flatten())
        mse_values.append(mse)
    return np.array(mse_values)

def classify_images(mse_values, threshold):
    return (mse_values > threshold).astype(int)

def visualize_hidden_space(features, targets, save_path=None):
    """Visualize hidden space features using t-SNE"""
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    df = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'label': targets
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='x', y='y',
        hue='label',
        palette=sns.color_palette("hsv", 2),
        data=df,
        legend="full",
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Hidden Space Clusters')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def classify(autoencoder, test_loader, threshold, save_dir=None, console=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_targets = []
    all_preds = []
    all_mse_values = []
    all_features = []
    
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if data is None:
                continue

            original_images = data.to(device)
            reconstructed_images, features, _ = autoencoder(original_images)
            
            all_features.append(features.cpu().numpy())
            
            mse_values = calculate_mse(reconstructed_images, original_images)
            predictions = classify_images(mse_values, threshold)

            all_preds = np.hstack((all_preds, predictions))
            all_targets = np.hstack((all_targets, targets.numpy()))
            all_mse_values = np.hstack((all_mse_values, mse_values))

            if i == 0 and save_dir:
                batch = data[:16]
                batch_grid = torchvision.utils.make_grid(batch, nrow=4)
                batch_grid = batch_grid.permute(1, 2, 0).cpu().numpy()
                plt.imshow(batch_grid)
                plt.axis('off')
                plt.suptitle('Original Test Batch')
                plt.savefig(os.path.join(save_dir, 'original_test_batch.png'), dpi=300)
                plt.close()

                batch = reconstructed_images[:16]
                batch_grid = torchvision.utils.make_grid(batch, nrow=4)
                batch_grid = batch_grid.permute(1, 2, 0).cpu().numpy()
                plt.imshow(batch_grid)
                plt.axis('off')
                plt.suptitle('Reconstructed Test Batch')
                plt.savefig(os.path.join(save_dir, 'rec_test_batch.png'), dpi=300)
                plt.close()

    if len(all_features) > 0:
        all_features = np.concatenate(all_features, axis=0)
        visualize_hidden_space(
            all_features, 
            all_targets, 
            save_path=os.path.join(save_dir, 'hidden_space_clusters.png') if save_dir else None
        )

    TP = np.sum((all_targets == 1) & (all_preds == 1))
    TN = np.sum((all_targets == 0) & (all_preds == 0))
    FP = np.sum((all_targets == 0) & (all_preds == 1))
    FN = np.sum((all_targets == 1) & (all_preds == 0))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

    if console:
        console.print(f'True Positive Rate (TPR): {TPR:.2f}', style='bold red')
        console.print(f'True Negative Rate (TNR): {TNR:.2f}', style='bold red')

    accuracy = accuracy_score(all_targets, all_preds)
    fpr, tpr, _ = roc_curve(all_targets, all_mse_values)
    roc_auc = AUC(fpr, tpr)

    if console:
        console.print(f'Accuracy: {accuracy:.2f}', style='bold green')
        console.print(f'AUC: {roc_auc:.2f}', style='bold green')

    if save_dir:
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
        plt.close()

    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'tpr': TPR,
        'tnr': TNR,
        'fpr': fpr,
        'tpr_curve': tpr
    }

def evaluate_mnad(model, m_items_test, test_loader, alpha=0.6, threshold=0.01, console=None):
    """Evaluation function specifically for MNAD model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    loss_func_mse = nn.MSELoss(reduction='none')
    
    all_targets = []
    all_psnr = []
    all_feature_dist = []
    all_features = []
    
    for k, (imgs, targets) in enumerate(test_loader):
        imgs = imgs.to(device)
        targets = targets.numpy()
        
        outputs, feas, updated_feas, m_items_test, _, _, compactness_loss = model.forward(
            imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        
        mse_feas = compactness_loss.item()
        point_sc = point_score(outputs, imgs)
        
        if point_sc < threshold:
            query = F.normalize(feas, dim=1)
            query = query.permute(0, 2, 3, 1)
            m_items_test = model.memory.update(query, m_items_test, False)
        
        for i in range(imgs.shape[0]):
            all_psnr.append(psnr(mse_imgs))
            all_feature_dist.append(mse_feas)
            all_targets.append(targets[i])
            all_features.append(feas[i].detach().cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_psnr = np.array(all_psnr)
    all_feature_dist = np.array(all_feature_dist)
    
    anomaly_scores = score_sum(
        anomaly_score_list(all_psnr),
        anomaly_score_list_inv(all_feature_dist),
        alpha
    )
    
    assert len(anomaly_scores) == len(all_targets), \
        f"Shape mismatch: scores {len(anomaly_scores)} vs targets {len(all_targets)}"
    
    try:
        auc_score = roc_auc_score(all_targets, anomaly_scores)
    except ValueError as e:
        console.print(f"[bold red]AUC calculation failed: {e}[/]")
        auc_score = 0.5
    
    if console:
        console.print(f'MNAD Evaluation', style='bold blue')
        console.print(f'AUC: {auc_score*100:.2f}%', style='bold green')
        console.print(f'Shapes - Scores: {np.array(anomaly_scores).shape}, Targets: {all_targets.shape}')
    
    if len(all_features) > 0:
        try:
            all_features = np.stack(all_features)
            visualize_hidden_space(
                all_features,
                all_targets,
                save_path='./figs/mnad_hidden_space.png'
            )
        except Exception as e:
            console.print(f"[bold red]Hidden space visualization failed: {e}[/]")
    
    return {
        'auc': auc_score,
        'anomaly_scores': anomaly_scores,
        'targets': all_targets,
        'memory': m_items_test
    }

@hydra.main(version_base=None)
def main(cfg):
    console = Console(record=True)
    model = utils.load_model(cfg)
    model.cuda()
    model.eval()
    m_items = torch.load(os.path.join(cfg.outdir, cfg.exp_name, 'memory_keys.pt'))

    data_list = cfg.dataset.test_lists
    test_loader = get_test_loader(cfg, data_list[0])
    results = evaluate_mnad(
        model,
        m_items,
        test_loader,
        alpha=0.6,
        threshold=0.0061,
        console=console,
    )

    return results

if __name__ == '__main__':
    main()