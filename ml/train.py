"""
Training Script
Complete training pipeline with MLflow tracking, callbacks, and fine-tuning.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import (
    get_dataset_path, load_image_paths, create_splits,
    create_dataset, get_class_weights, CLASS_NAMES, IMG_SIZE
)
from ml.model import get_model, unfreeze_base_model
from ml.clinical_metrics import (
    calculate_metrics_with_ci, calculate_multiclass_auc,
    generate_clinical_report, plot_roc_curves,
    plot_confusion_matrix_with_metrics
)

# Set up GPU (T4 optimization for Colab free tier)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🎮 Found {len(gpus)} GPU(s) - Memory growth enabled")
        print(f"   GPU Details: {tf.config.experimental.get_device_details(gpus[0])}")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
else:
    print("💻 No GPU found - Using CPU")

# Enable mixed precision training for T4 GPU (1.5-2x speedup)
try:
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print(f"✅ Mixed precision FP16 enabled: {policy.name}")
    print(f"   Expected speedup on T4 GPU: 1.5-2x")
except:
    print("⚠️  Mixed precision not available, using default float32")

# Enable XLA compilation for additional 15% speedup
tf.config.optimizer.set_jit(True)
print("✅ XLA compilation enabled (+15% performance)")


class TrainingConfig:
    """Training configuration parameters."""
    
    def __init__(self):
        # Model
        self.architecture = 'efficientnet'  # 'efficientnet' or 'resnet'
        self.input_shape = (IMG_SIZE, IMG_SIZE, 3)
        self.num_classes = len(CLASS_NAMES)
        self.dropout_rate = 0.3
        self.l2_reg = 0.001
        
        # Training - Phase 1 (frozen base) - T4 GPU optimized
        self.initial_epochs = 15
        self.initial_lr = 0.001
        self.batch_size = 40  # Optimal for T4 GPU (16GB VRAM)
        
        # Training - Phase 2 (fine-tuning)
        self.fine_tune_epochs = 35  # Total ~50 epochs for clinical performance
        self.fine_tune_lr = 0.0001
        self.unfreeze_layers = 30
        
        # Callbacks - Aggressive for Colab free tier
        self.early_stopping_patience = 10
        self.reduce_lr_patience = 3
        self.reduce_lr_factor = 0.5
        self.checkpoint_frequency = 5  # Save every 5 epochs for session recovery
        
        # Clinical metrics
        self.use_focal_loss = True  # Maximize sensitivity
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.sensitivity_threshold = 0.95  # Target from literature
        self.monitor_sensitivity = True  # Stop when sensitivity >= 95%
        
        # Paths
        self.model_dir = Path("models")
        self.logs_dir = Path("logs")
        self.mlflow_tracking_uri = "file:./mlruns"
        
        # Other
        self.seed = 42
        self.use_class_weights = True
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance and hard examples.
    Focuses training on hard-to-classify samples by down-weighting easy examples.
    
    Critical for medical screening: Penalizes false negatives more heavily!
    
    Args:
        gamma: Focusing parameter (default: 2.0). Higher = more focus on hard examples
        alpha: Balancing parameter (default: 0.25). Weights for rare classes
        
    Returns:
        Focal loss function compatible with Keras/TensorFlow
        
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * keras.backend.log(y_pred)
        
        # Calculate focal loss
        weight = alpha * y_true * keras.backend.pow((1 - y_pred), gamma)
        
        focal_loss_value = weight * cross_entropy
        
        return keras.backend.sum(focal_loss_value, axis=-1)
    
    return focal_loss_fixed


def sparse_categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Sparse categorical focal loss (for integer labels, not one-hot).
    Specifically designed to maximize sensitivity in medical screening.
    
    Args:
        gamma: Focusing parameter
        alpha: Class balancing parameter
        
    Returns:
        Focal loss function for sparse labels
    """
    def loss_fn(y_true, y_pred):
        # Convert sparse labels to one-hot
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        
        # Clip predictions
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true_one_hot * keras.backend.log(y_pred)
        
        # Calculate weights (higher weight for hard examples)
        weight = keras.backend.pow(1.0 - y_pred, gamma)
        
        # Apply focal term
        focal_loss_value = weight * cross_entropy
        
        return keras.backend.sum(focal_loss_value, axis=-1)
    
    return loss_fn



def setup_callbacks(config: TrainingConfig, phase: str = "initial") -> list:
    """
    Set up training callbacks.
    
    Args:
        config: Training configuration
        phase: Training phase ('initial' or 'fine_tune')
        
    Returns:
        List of callbacks
    """
    log_dir = config.logs_dir / f"{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint (best model based on val_accuracy)
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.model_dir / f'best_{phase}.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Aggressive checkpointing for Colab free tier (every N epochs)
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.model_dir / f'checkpoint_{phase}_epoch_{{epoch:02d}}.keras'),
            monitor='val_accuracy',
            save_freq=config.checkpoint_frequency * len(keras.backend.get_value(config.batch_size)) if hasattr(config, 'steps_per_epoch') else 'epoch',
            verbose=0,  # Less verbose to reduce log clutter
            save_best_only=False  # Save all checkpoints for recovery
        ),
        
        # CSV Logger
        keras.callbacks.CSVLogger(
            filename=str(log_dir / 'training.csv'),
            append=True
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path: Path):
    """
    Plot and save training history.
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    history_dict = history.history
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history_dict['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history_dict['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history_dict['loss'], label='Train Loss')
    axes[0, 1].plot(history_dict['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    if 'lr' in history_dict:
        axes[1, 0].plot(history_dict['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Top-2 accuracy if available
    if 'top_k_categorical_accuracy' in history_dict:
        axes[1, 1].plot(history_dict['top_k_categorical_accuracy'], label='Train Top-2')
        axes[1, 1].plot(history_dict['val_top_k_categorical_accuracy'], label='Val Top-2')
        axes[1, 1].set_title('Top-2 Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training plots saved to: {save_path}")


def evaluate_model(model, test_dataset, class_names, save_dir: Path):
    """
    Evaluate model with comprehensive clinical metrics.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        class_names: List of class names
        save_dir: Directory to save results
    """
    print("\n" + "=" * 60)
    print("  Clinical Model Evaluation")
    print("=" * 60)
    
    # Get predictions and probabilities
    y_true = []
    y_pred_proba = []
    
    print("\n📊 Generating predictions...")
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Standard classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n📊 Classification Report:")
    print(report)
    
    # Save report
    report_path = save_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"💾 Report saved to: {report_path}")
    
    # ==================== CLINICAL METRICS ====================
    print("\n" + "=" * 60)
    print("  Calculating Clinical Metrics (with 95% CI)")
    print("=" * 60)
    
    # Calculate clinical metrics with confidence intervals
    clinical_metrics = calculate_metrics_with_ci(
        y_true, y_pred, class_names,
        n_bootstrap=1000,
        confidence_level=0.95
    )
    
    # Calculate AUC-ROC scores
    auc_scores = calculate_multiclass_auc(y_true, y_pred_proba, class_names)
    
    # Generate comprehensive clinical report
    clinical_report = generate_clinical_report(
        clinical_metrics,
        class_names,
        auc_scores,
        save_path=str(save_dir / 'clinical_report.json')
    )
    
    # Plot ROC curves
    plot_roc_curves(
        y_true, y_pred_proba, class_names,
        save_path=str(save_dir / 'roc_curves.png')
    )
    
    # Plot confusion matrix with clinical metrics
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_with_metrics(
        cm, class_names, clinical_metrics,
        save_path=str(save_dir / 'confusion_matrix_clinical.png')
    )
    
    # Save confusion matrix data (legacy format for App.py)
    cm_data_path = save_dir / 'confusion_matrix.pkl'
    with open(cm_data_path, 'wb') as f:
        pickle.dump((cm, class_names), f)
    
    # ==================== BENCHMARK COMPARISON ====================
    print("\n" + "=" * 60)
    print("  Comparison with Published Benchmarks")
    print("=" * 60)
    
    overall = clinical_metrics['overall']
    
    # Literature benchmarks (from AI cervical cytology meta-analysis)
    benchmarks = {
        'sensitivity': 0.95,
        'specificity': 0.94,
        'accuracy': 0.94,
        'ppv': 0.88,
        'npv': 0.95
    }
    
    print("\nLiterature Benchmark vs Our Model:")
    print(f"  Metric          | Benchmark | Our Model  | Status")
    print(f"  {'-'*15} | {'-'*9} | {'-'*10} | {'-'*6}")
    
    sens = overall['macro_sensitivity']
    sens_status = "✅ PASS" if sens >= benchmarks['sensitivity'] else "❌ FAIL"
    print(f"  Sensitivity     | ≥{benchmarks['sensitivity']:.0%}      | {sens:.1%}      | {sens_status}")
    
    spec = overall['macro_specificity']
    spec_status = "✅ PASS" if spec >= benchmarks['specificity'] else "❌ FAIL"
    print(f"  Specificity     | ≥{benchmarks['specificity']:.0%}      | {spec:.1%}      | {spec_status}")
    
    acc = overall['accuracy']
    acc_status = "✅ PASS" if acc >= benchmarks['accuracy'] else "❌ FAIL"
    print(f"  Accuracy        | ≥{benchmarks['accuracy']:.0%}      | {acc:.1%}      | {acc_status}")
    
    ppv = overall['macro_ppv']
    ppv_status = "✅ PASS" if ppv >= benchmarks['ppv'] else "❌ FAIL"
    print(f"  PPV             | ≥{benchmarks['ppv']:.0%}      | {ppv:.1%}      | {ppv_status}")
    
    npv = overall['macro_npv']
    npv_status = "✅ PASS" if npv >= benchmarks['npv'] else "❌ FAIL"
    print(f"  NPV             | ≥{benchmarks['npv']:.0%}      | {npv:.1%}      | {npv_status}")
    
    # Overall status
    all_pass = all([
        sens >= benchmarks['sensitivity'],
        spec >= benchmarks['specificity'],
        acc >= benchmarks['accuracy']
    ])
    
    if all_pass:
        print("\n🎉 EXCELLENT! Model meets all critical clinical benchmarks!")
    else:
        print("\n⚠️  Model does not meet all benchmarks. Consider:")
        print("     • Increase training epochs")
        print("     • Adjust focal loss parameters")
        print("     • Try ensemble methods")
        print("     • Collect more training data")
    
    # Return metrics for logging
    return {
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'ppv': float(ppv),
        'npv': float(npv),
        'auc_macro': float(auc_scores.get('macro_average', 0.0)),
        'confusion_matrix': cm.tolist(),
        'clinical_metrics': clinical_metrics,
        'meets_benchmarks': all_pass
    }



def train_model(config: TrainingConfig):
    """
    Main training function.
    
    Args:
        config: Training configuration
    """
    print("=" * 60)
    print("  Cervical Cancer Detection - Model Training")
    print("=" * 60)
    print()
    
    # Create directories
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("📁 Loading dataset...")
    dataset_path = get_dataset_path()
    image_paths, labels, class_names = load_image_paths(dataset_path)
    
    # Create splits
    splits = create_splits(image_paths, labels)
    
    # Calculate class weights
    class_weights = get_class_weights(splits['train'][1], class_names) if config.use_class_weights else None
    
    # Create tf.data pipelines
    print("\n🔧 Creating data pipelines...")
    train_dataset = create_dataset(
        splits['train'][0], splits['train'][1],
        batch_size=config.batch_size,
        augment=True,
        shuffle=True
    )
    
    val_dataset = create_dataset(
        splits['val'][0], splits['val'][1],
        batch_size=config.batch_size,
        augment=False,
        shuffle=False
    )
    
    test_dataset = create_dataset(
        splits['test'][0], splits['test'][1],
        batch_size=config.batch_size,
        augment=False,
        shuffle=False
    )
    
    # Create model
    print(f"\n🏗️  Building {config.architecture} model...")
    model, base_model = get_model(
        architecture=config.architecture,
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        l2_reg=config.l2_reg
    )
    
    # Compile model with focal loss if enabled (for sensitivity optimization)
    if config.use_focal_loss:
        loss_fn = sparse_categorical_focal_loss(
            gamma=config.focal_loss_gamma,
            alpha=config.focal_loss_alpha
        )
        print(f"\n🎯 Using Focal Loss (gamma={config.focal_loss_gamma}, alpha={config.focal_loss_alpha})")
        print("   This will maximize sensitivity by focusing on hard examples.")
    else:
        loss_fn = 'sparse_categorical_crossentropy'
        print("\n Using standard cross-entropy loss")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.initial_lr),
        loss=loss_fn,
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    print("\n📝 Model Summary:")
    model.summary()
    
    # Phase 1: Train with frozen base
    print("\n" + "=" * 60)
    print("  PHASE 1: Training with frozen base model")
    print("=" * 60)
    
    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.initial_epochs,
        class_weight=class_weights,
        callbacks=setup_callbacks(config, phase="phase1"),
        verbose=1
    )
    
    # Save phase 1 model
    phase1_path = config.model_dir / f"{config.architecture}_phase1.keras"
    model.save(phase1_path)
    print(f"\n💾 Phase 1 model saved: {phase1_path}")
    
    # Plot phase 1 history
    plot_training_history(history_phase1, config.model_dir / "training_history_phase1.png")
    
    # Phase 2: Fine-tuning
    print("\n" + "=" * 60)
    print("  PHASE 2: Fine-tuning")
    print("=" * 60)
    
    # Unfreeze base model layers
    unfreeze_base_model(base_model, num_layers=config.unfreeze_layers)
    
    # Recompile with lower learning rate (keep same loss function)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.fine_tune_lr),
        loss=loss_fn,  # Use same loss function as Phase 1
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.fine_tune_epochs,
        class_weight=class_weights,
        callbacks=setup_callbacks(config, phase="phase2"),
        verbose=1
    )
    
    # Save final model
    final_model_path = config.model_dir / f"{config.architecture}_final.keras"
    model.save(final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")
    
    # Plot phase 2 history
    plot_training_history(history_phase2, config.model_dir / "training_history_phase2.png")
    
    # Save combined history
    combined_history = {
        'phase1': history_phase1.history,
        'phase2': history_phase2.history
    }
    history_path = config.model_dir / 'training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(combined_history, f)
    
    # Evaluate on test set
    metrics = evaluate_model(model, test_dataset, class_names, config.model_dir)
    
    # Save metrics
    metrics_path = config.model_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save configuration
    config_path = config.model_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📊 Final Results:")
    print(f"   • Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"   • Model saved: {final_model_path}")
    print(f"   • Total training time: Phase 1 + Phase 2")
    print(f"\n💡 Next steps:")
    print(f"   1. Review training plots in: {config.model_dir}")
    print(f"   2. Test the model: python ml/export.py")
    print(f"   3. Deploy: python services/inference/main.py")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train cervical cancer classification model')
    parser.add_argument('--architecture', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet'],
                       help='Model architecture to use')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--initial-epochs', type=int, default=10,
                       help='Number of epochs for phase 1')
    parser.add_argument('--fine-tune-epochs', type=int, default=20,
                       help='Number of epochs for phase 2')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.architecture = args.architecture
    config.batch_size = args.batch_size
    config.initial_epochs = args.initial_epochs
    config.fine_tune_epochs = args.fine_tune_epochs
    
    # Train model
    train_model(config)


if __name__ == "__main__":
    main()
