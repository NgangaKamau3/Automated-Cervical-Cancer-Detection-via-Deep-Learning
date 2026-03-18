"""
Model Export Module
Handles model export to different formats and validation.
"""

import os
import sys
import argparse
import json
import tensorflow as tf
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def export_keras_model(model_path: str, export_dir: str = "models/export"):
    """
    Export Keras model in SavedModel format.
    
    Args:
        model_path: Path to the trained .keras model
        export_dir: Directory to save exported model
    """
    print(f"📦 Exporting Keras model from: {model_path}")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Create export directory
    export_path = Path(export_dir) / "saved_model"
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Export as SavedModel
    model.save(str(export_path), save_format='tf')
    print(f"✅ SavedModel exported to: {export_path}")
    
    return str(export_path)


def export_tflite_model(model_path: str, export_dir: str = "models/export",
                       quantize: bool = False):
    """
    Export model to TensorFlow Lite format.
    
    Args:
        model_path: Path to the trained .keras model
        export_dir: Directory to save exported model
        quantize: Whether to apply post-training quantization
    """
    print(f"📱 Exporting TFLite model from: {model_path}")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("   🔧 Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        suffix = "_quantized"
    else:
        suffix = ""
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    export_path = Path(export_dir) / f"model{suffix}.tflite"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(export_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size_mb = export_path.stat().st_size / (1024 * 1024)
    print(f"✅ TFLite model exported to: {export_path} ({file_size_mb:.2f} MB)")
    
    return str(export_path)


def save_model_metadata(model_path: str, export_dir: str = "models/export"):
    """
    Save model metadata including class labels and preprocessing config.
    
    Args:
        model_path: Path to the trained model
        export_dir: Directory to save metadata
    """
    print("📝 Saving model metadata...")
    
    # Load dataset info
    dataset_info_path = Path("models/dataset_info.json")
    if dataset_info_path.exists():
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
    
    # Load training config
    config_path = Path("models/training_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            training_config = json.load(f)
    else:
        training_config = {}
    
    # Load test metrics
    metrics_path = Path("models/test_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            test_metrics = json.load(f)
    else:
        test_metrics = {}
    
    # Combine metadata
    metadata = {
        'model_path': str(model_path),
        'class_names': dataset_info.get('class_names', []),
        'num_classes': dataset_info.get('num_classes', 5),
        'input_size': dataset_info.get('image_size', 224),
        'preprocessing': {
            'resize': dataset_info.get('image_size', 224),
            'normalization': 'divide_by_255',
            'color_mode': 'rgb'
        },
        'performance': {
            'test_accuracy': test_metrics.get('accuracy', None),
            'precision': test_metrics.get('precision', None),
            'recall': test_metrics.get('recall', None),
            'f1': test_metrics.get('f1', None)
        },
        'training': training_config,
        'version': '1.0.0',
        'framework': 'tensorflow',
        'framework_version': tf.__version__
    }
    
    # Save metadata
    export_path = Path(export_dir) / "model_metadata.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(export_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {export_path}")
    
    return metadata


def validate_exported_model(original_model_path: str, exported_model_path: str,
                           test_images_dir: str = None):
    """
    Validate that exported model produces same predictions as original.
    
    Args:
        original_model_path: Path to original .keras model
        exported_model_path: Path to exported SavedModel
        test_images_dir: Optional directory with test images
    """
    print("\n🧪 Validating exported model...")
    
    # Load original model
    original_model = tf.keras.models.load_model(original_model_path)
    
    # Load exported model
    exported_model = tf.keras.models.load_model(exported_model_path)
    
    # Create dummy test input
    dummy_input = tf.random.normal((5, 224, 224, 3))
    
    # Get predictions
    original_preds = original_model.predict(dummy_input, verbose=0)
    exported_preds = exported_model.predict(dummy_input, verbose=0)
    
    # Check if predictions match
    max_diff = np.max(np.abs(original_preds - exported_preds))
    mean_diff = np.mean(np.abs(original_preds - exported_preds))
    
    print(f"   • Max prediction difference: {max_diff:.6f}")
    print(f"   • Mean prediction difference: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("   ✅ Exported model predictions match original!")
        return True
    else:
        print(f"   ⚠️  Warning: Predictions differ by {max_diff:.6f}")
        return False


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export trained model to different formats')
    parser.add_argument('--model', type=str, default='models/efficientnet_final.keras',
                       help='Path to trained model')
    parser.add_argument('--export-dir', type=str, default='models/export',
                       help='Directory to save exported models')
    parser.add_argument('--tflite', action='store_true',
                       help='Export to TFLite format')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply quantization to TFLite model')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate exported model')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Model Export Tool")
    print("=" * 60)
    print()
    
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please train a model first using: python ml/train.py")
        sys.exit(1)
    
    export_dir = args.export_dir
    
    # Export SavedModel (always)
    saved_model_path = export_keras_model(str(model_path), export_dir)
    
    # Export TFLite if requested
    if args.tflite:
        tflite_path = export_tflite_model(str(model_path), export_dir, quantize=args.quantize)
    
    # Save metadata
    metadata = save_model_metadata(str(model_path), export_dir)
    
    # Validate
    if args.validate:
        validate_exported_model(str(model_path), saved_model_path)
    
    print("\n" + "=" * 60)
    print("✅ Export Complete!")
    print("=" * 60)
    print("\n Exported models:")
    print(f"   • SavedModel: {saved_model_path}")
    if args.tflite:
        print(f"   • TFLite: {tflite_path}")
    print(f"   • Metadata: {export_dir}/model_metadata.json")
    
    if metadata.get('performance', {}).get('test_accuracy'):
        print("\n Model Performance:")
        print(f"   • Test Accuracy: {metadata['performance']['test_accuracy'] * 100:.2f}%")
    
    print("\n Next steps:")
    print(" 1. Test the exported model in the inference service")
    print(" 2. Deploy using Docker: docker build -t cervical-cancer-detection .")


if __name__ == "__main__":
    main()
