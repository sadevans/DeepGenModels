import warnings
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import utils
from rich.console import Console
from sklearn.metrics import (auc, mean_squared_error, precision_recall_curve,
                           roc_curve, confusion_matrix, accuracy_score)
from data import get_test_loader
import os

warnings.filterwarnings('ignore')

class MNADEvaluator:
    def __init__(self, cfg, console):
        self.cfg = cfg
        self.console = console
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = utils.load_model(cfg).to(self.device)
        self.model.eval()
        
        self.memory_items = F.normalize(
            torch.rand((cfg.model.memory_size, cfg.model.key_dim), dtype=torch.float),
            dim=1
        ).to(self.device)
        
        os.makedirs('./figs', exist_ok=True)

    def calculate_mse(self, outputs, inputs):
        return F.mse_loss((outputs + 1) / 2, (inputs + 1) / 2, reduction='none').mean(dim=[1, 2, 3]).cpu().numpy()

    def find_optimal_threshold(self, proliv_loader, non_proliv_loader):
        self.console.print("[bold]Finding optimal threshold...[/bold]")
        
        proliv_scores = self._calculate_scores(proliv_loader)
        non_proliv_scores = self._calculate_scores(non_proliv_loader)
        
        all_scores = np.concatenate([proliv_scores, non_proliv_scores])
        true_labels = np.concatenate([np.ones(len(proliv_scores)), np.zeros(len(non_proliv_scores))])
        
        fpr, tpr, thresholds = roc_curve(true_labels, all_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        roc_auc = auc(fpr, tpr)
        
        self._plot_threshold_results(proliv_scores, non_proliv_scores, 
                                   fpr, tpr, roc_auc, optimal_threshold, optimal_idx)
        
        return optimal_threshold, roc_auc

    def _calculate_scores(self, loader):
        scores = []
        with torch.no_grad():
            for data in loader:
                if data is None:
                    continue
                data = data.to(self.device)
                
                outputs, *_ = self.model(data, self.memory_items, train=False)
                batch_scores = self.calculate_mse(outputs, data)
                scores.extend(batch_scores)
        return np.array(scores)

    def _plot_threshold_results(self, proliv_scores, non_proliv_scores,
                              fpr, tpr, roc_auc, optimal_threshold, optimal_idx):
        plt.figure(figsize=(15, 5))
        
        # Score distribution
        plt.subplot(1, 3, 1)
        plt.hist(proliv_scores, bins=50, alpha=0.7, label='Spill', color='red')
        plt.hist(non_proliv_scores, bins=50, alpha=0.7, label='Non-Spill', color='green')
        plt.axvline(optimal_threshold, color='black', linestyle='dashed', 
                    linewidth=2, label=f'Threshold: {optimal_threshold:.4f}')
        plt.xlabel('MSE Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Score Distribution')

        # ROC curve
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

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(
            np.concatenate([np.ones(len(proliv_scores)), np.zeros(len(non_proliv_scores))]),
            np.concatenate([proliv_scores, non_proliv_scores])
        )
        plt.subplot(1, 3, 3)
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig('./figs/threshold_analysis.png', dpi=300)
        plt.close()

    def evaluate_test_set(self, test_loader, threshold):
        """Evaluate model performance on test set"""
        self.console.print("[bold]Evaluating on test set...[/bold]")
        
        all_scores = []
        all_labels = []
        all_outputs = []
        
        i = -1
        with torch.no_grad():
            for data, labels in test_loader:
                i += 1
                data = data.to(self.device)
                labels = labels.cpu().numpy()
                
                outputs, *_ = self.model(data, self.memory_items, train=False)
                batch_scores = self.calculate_mse(outputs, data)
                
                all_scores.extend(batch_scores)
                all_labels.extend(labels)
                all_outputs.extend((batch_scores > threshold).astype(int))

                if i == 0:
                    batch = data[:16]
                    batch_grid = torchvision.utils.make_grid(batch, nrow=4)
                    batch_grid = batch_grid.permute(1, 2, 0).cpu().numpy()
                    plt.imshow(batch_grid)
                    plt.axis('off')
                    plt.suptitle('Original Test Batch')
                    plt.savefig(os.path.join('./figs/', 'original_test_batch.png'), dpi=300)
                    plt.close()

                    batch = outputs[:16]
                    batch_grid = torchvision.utils.make_grid(batch, nrow=4)
                    batch_grid = batch_grid.permute(1, 2, 0).cpu().numpy()
                    plt.imshow(batch_grid)
                    plt.axis('off')
                    plt.suptitle('Reconstructed Test Batch')
                    plt.savefig(os.path.join('./figs/', 'rec_test_batch.png'), dpi=300)
                    plt.close()
        
        tn, fp, fn, tp = confusion_matrix(all_labels, all_outputs).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        accuracy = accuracy_score(all_labels, all_outputs)
        
        self.console.print(f"[bold green]Test Results:[/bold green]")
        self.console.print(f"True Positive Rate (TPR): {tpr:.4f}")
        self.console.print(f"True Negative Rate (TNR): {tnr:.4f}")
        self.console.print(f"Accuracy: {accuracy:.4f}")
        
        self._plot_confusion_matrix(all_labels, all_outputs)
        self._plot_test_roc_curve(all_labels, all_scores)

        
        return tpr, tnr, accuracy
    
    def _plot_test_roc_curve(self, true_labels, scores):
        """Plot ROC curve for test set"""
        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Test Set)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('./figs/test_roc_curve.png', dpi=300)
        plt.close()

    def _plot_confusion_matrix(self, true_labels, pred_labels):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['Non-Spill', 'Spill']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('./figs/confusion_matrix.png', dpi=300)
        plt.close()


@hydra.main(version_base=None)
def main(cfg):
    console = Console(record=True)
    evaluator = MNADEvaluator(cfg, console)
    
    proliv_loader = get_test_loader(cfg, cfg.dataset.val_lists[0])
    non_proliv_loader = get_test_loader(cfg, cfg.dataset.train_lists[0])
    test_loader = get_test_loader(cfg, cfg.dataset.test_lists[0])
    
    threshold, roc_auc = evaluator.find_optimal_threshold(proliv_loader, non_proliv_loader)
    console.print(f"[bold]Optimal Threshold:[/bold] {threshold:.4f}")
    console.print(f"[bold]ROC AUC:[/bold] {roc_auc:.4f}")
    
    tpr, tnr, accuracy = evaluator.evaluate_test_set(test_loader, threshold)

if __name__ == '__main__':
    main()