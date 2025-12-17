"""
Clinical Metrics Module
Calculates medical-grade metrics for cervical cancer detection.
Focuses on sensitivity, specificity, PPV, NPV, and AUC-ROC with confidence intervals.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, cohen_kappa_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json


def calculate_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: List[str]) -> Dict:
    """
    Calculate comprehensive clinical metrics.
    
    Args:
        y_true: True labels (1D array of class indices)
        y_pred: Predicted labels (1D array of class indices)
        class_names: List of class names
        
    Returns:
        Dictionary with clinical metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    
    metrics = {
        'overall': {},
        'per_class': {}
    }
    
    # Overall metrics
    accuracy = np.mean(y_true == y_pred)
    metrics['overall']['accuracy'] = float(accuracy)
    metrics['overall']['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        # Binary classification: class i vs rest
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Sensitivity (Recall): TP / (TP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        # Specificity: TN / (TN + FP)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        
        # PPV (Precision): TP / (TP + FP)
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        
        # NPV: TN / (TN + FN)
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        
        # F1 Score
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
        
        metrics['per_class'][class_name] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'f1_score': float(f1),
            'support': int(np.sum(y_true == i)),
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN)
        }
    
    # Macro-averaged metrics (average across classes)
    sensitivities = [metrics['per_class'][name]['sensitivity'] for name in class_names]
    specificities = [metrics['per_class'][name]['specificity'] for name in class_names]
    ppvs = [metrics['per_class'][name]['ppv'] for name in class_names]
    npvs = [metrics['per_class'][name]['npv'] for name in class_names]
    
    metrics['overall']['macro_sensitivity'] = float(np.mean(sensitivities))
    metrics['overall']['macro_specificity'] = float(np.mean(specificities))
    metrics['overall']['macro_ppv'] = float(np.mean(ppvs))
    metrics['overall']['macro_npv'] = float(np.mean(npvs))
    
    # Min sensitivity (weakest class - CRITICAL for medical screening)
    metrics['overall']['min_sensitivity'] = float(np.min(sensitivities))
    metrics['overall']['weakest_class'] = class_names[np.argmin(sensitivities)]
    
    return metrics


def calculate_metrics_with_ci(y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: List[str],
                               n_bootstrap: int = 1000,
                               confidence_level: float = 0.95) -> Dict:
    """
    Calculate clinical metrics with confidence intervals using bootstrapping.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (default: 95%)
        
    Returns:
        Dictionary with metrics and confidence intervals
    """
    n_samples = len(y_true)
    alpha = 1 - confidence_level
    
    # Calculate point estimates
    metrics = calculate_clinical_metrics(y_true, y_pred, class_names)
    
    # Bootstrap for confidence intervals
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        boot_metrics = calculate_clinical_metrics(y_true_boot, y_pred_boot, class_names)
        bootstrap_metrics.append(boot_metrics)
    
    # Calculate confidence intervals for overall metrics
    for metric_name in ['accuracy', 'macro_sensitivity', 'macro_specificity', 
                        'macro_ppv', 'macro_npv', 'cohen_kappa']:
        values = [m['overall'][metric_name] for m in bootstrap_metrics]
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        
        metrics['overall'][f'{metric_name}_ci'] = [float(lower), float(upper)]
    
    # Calculate confidence intervals for per-class metrics
    for class_name in class_names:
        for metric_name in ['sensitivity', 'specificity', 'ppv', 'npv', 'f1_score']:
            values = [m['per_class'][class_name][metric_name] for m in bootstrap_metrics]
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            
            metrics['per_class'][class_name][f'{metric_name}_ci'] = [float(lower), float(upper)]
    
    return metrics


def calculate_multiclass_auc(y_true: np.ndarray, y_pred_proba: np.ndarray,
                             class_names: List[str]) -> Dict:
    """
    Calculate AUC-ROC for multiclass classification (one-vs-rest).
    
    Args:
        y_true: True labels (1D array)
        y_pred_proba: Predicted probabilities (2D array: samples x classes)
        class_names: List of class names
        
    Returns:
        Dictionary with AUC scores
    """
    n_classes = len(class_names)
    auc_scores = {}
    
    for i, class_name in enumerate(class_names):
        # Binary: class i vs rest
        y_true_binary = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        try:
            auc_score = roc_auc_score(y_true_binary, y_score)
            auc_scores[class_name] = float(auc_score)
        except ValueError:
            # Handle case where class is not in test set
            auc_scores[class_name] = None
    
    # Macro average (mean of per-class AUC)
    valid_aucs = [v for v in auc_scores.values() if v is not None]
    auc_scores['macro_average'] = float(np.mean(valid_aucs)) if valid_aucs else None
    
    return auc_scores


def plot_roc_curves(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   class_names: List[str], save_path: str = None):
    """
    Plot ROC curves for all classes.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        save_path: Path to save figure (optional)
    """
    n_classes = len(class_names)
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        except ValueError:
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves - Multiclass Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.close()


def generate_clinical_report(metrics: Dict, class_names: List[str],
                             auc_scores: Dict = None,
                             save_path: str = "clinical_report.json"):
    """
    Generate a comprehensive clinical validation report.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics_with_ci
        class_names: List of class names
        auc_scores: Optional AUC scores
        save_path: Path to save report
    """
    report = {
        'clinical_performance': {
            'overall_metrics': metrics['overall'],
            'per_class_metrics': metrics['per_class']
        }
    }
    
    if auc_scores:
        report['auc_roc'] = auc_scores
    
    # Clinical interpretation
    overall = metrics['overall']
    interpretation = []
    
    # Check against clinical benchmarks
    if overall['macro_sensitivity'] >= 0.95:
        interpretation.append("✅ Sensitivity meets clinical benchmark (≥95%)")
    else:
        interpretation.append(f"⚠️ Sensitivity below benchmark: {overall['macro_sensitivity']:.1%} < 95%")
    
    if overall['macro_specificity'] >= 0.94:
        interpretation.append("✅ Specificity meets clinical benchmark (≥94%)")
    else:
        interpretation.append(f"⚠️ Specificity below benchmark: {overall['macro_specificity']:.1%} < 94%")
    
    if overall['accuracy'] >= 0.94:
        interpretation.append("✅ Accuracy meets clinical benchmark (≥94%)")
    else:
        interpretation.append(f"⚠️ Accuracy below benchmark: {overall['accuracy']:.1%} < 94%")
    
    # Flag weakest class
    min_sens = overall['min_sensitivity']
    weak_class = overall['weakest_class']
    if min_sens < 0.90:
        interpretation.append(f"🔴 CRITICAL: {weak_class} has low sensitivity ({min_sens:.1%})")
    elif min_sens < 0.95:
        interpretation.append(f"⚠️ Warning: {weak_class} has borderline sensitivity ({min_sens:.1%})")
    
    report['clinical_interpretation'] = interpretation
    
    # Save report
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Clinical Report Generated: {save_path}")
    print("\nClinical Performance Summary:")
    print(f"  • Sensitivity: {overall['macro_sensitivity']:.1%} "
          f"(95% CI: {overall['macro_sensitivity_ci'][0]:.1%}-{overall['macro_sensitivity_ci'][1]:.1%})")
    print(f"  • Specificity: {overall['macro_specificity']:.1%} "
          f"(95% CI: {overall['macro_specificity_ci'][0]:.1%}-{overall['macro_specificity_ci'][1]:.1%})")
    print(f"  • Accuracy: {overall['accuracy']:.1%} "
          f"(95% CI: {overall['accuracy_ci'][0]:.1%}-{overall['accuracy_ci'][1]:.1%})")
    print(f"  • PPV: {overall['macro_ppv']:.1%}")
    print(f"  • NPV: {overall['macro_npv']:.1%}")
    print(f"\n  Weakest Class: {weak_class} (Sensitivity: {min_sens:.1%})")
    
    print("\n" + "\n".join(interpretation))
    
    return report


def plot_confusion_matrix_with_metrics(cm: np.ndarray, class_names: List[str],
                                       metrics: Dict, save_path: str = None):
    """
    Plot confusion matrix with clinical metrics annotated.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        metrics: Metrics dictionary
        save_path: Path to save figure
    """
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(cm_norm, annot=False, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Percentage'})
    
    # Add custom annotations with counts and percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percentage = cm_norm[i, j]
            
            # Highlight diagonal (correct predictions)
            color = 'white' if percentage > 0.5 else 'black'
            weight = 'bold' if i == j else 'normal'
            
            ax.text(j + 0.5, i + 0.5, f'{count}\n({percentage:.1%})',
                   ha='center', va='center', color=color, fontweight=weight,
                   fontsize=9)
    
    # Add sensitivity on the right
    for i, class_name in enumerate(class_names):
        sens = metrics['per_class'][class_name]['sensitivity']
        color = 'green' if sens >= 0.95 else ('orange' if sens >= 0.90 else 'red')
        ax.text(len(class_names) + 0.5, i + 0.5, f'{sens:.1%}',
               ha='center', va='center', fontsize=10, fontweight='bold',
               color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels
    ax.text(len(class_names) + 0.5, -0.5, 'Sensitivity',
           ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix with Clinical Metrics\n(Sensitivity shown on right)',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test with dummy data
    print("Testing clinical metrics module...")
    
    # Generate dummy predictions
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    class_names = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors (10%)
    error_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, len(error_indices))
    
    # Generate probabilities
    y_pred_proba = np.random.dirichlet(np.ones(n_classes) * 10, n_samples)
    
    # Calculate metrics
    print("\n1. Calculating metrics with confidence intervals...")
    metrics = calculate_metrics_with_ci(y_true, y_pred, class_names, n_bootstrap=100)
    
    print("\n2. Calculating AUC scores...")
    auc_scores = calculate_multiclass_auc(y_true, y_pred_proba, class_names)
    
    print("\n3. Generating clinical report...")
    report = generate_clinical_report(metrics, class_names, auc_scores)
    
    print("\n✅ Clinical metrics module test complete!")
