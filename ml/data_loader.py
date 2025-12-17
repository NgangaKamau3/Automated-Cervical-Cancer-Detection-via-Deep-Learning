"""
Data Loader Module
Handles dataset loading, preprocessing, augmentation, and tf.data pipeline creation.
"""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import json
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Image parameters
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Class mapping - SIPaKMeD dataset
CLASS_NAMES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate"
]


def get_dataset_path() -> str:
    """
    Get the dataset path from config file or environment variable.
    
    Returns:
        Path to the dataset directory
    """
    # Check config file first
    config_file = Path("config/dataset_path.txt")
    if config_file.exists():
        return config_file.read_text().strip()
    
    # Check environment variable
    if "SIPAKMED_DATASET_PATH" in os.environ:
        return os.environ["SIPAKMED_DATASET_PATH"]
    
    # Default location
    default_path = Path("data/raw")
    if default_path.exists():
        return str(default_path)
    
    raise FileNotFoundError(
        "Dataset path not found. Please run 'python scripts/download_dataset.py' first, "
        "or set the SIPAKMED_DATASET_PATH environment variable."
    )


def load_image_paths(dataset_path: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Load all image paths and labels from the dataset directory.
    
    Args:
        dataset_path: Path to the dataset root directory
        
    Returns:
        Tuple of (image_paths, labels, class_names)
    """
    dataset_path = Path(dataset_path)
    image_paths = []
    labels = []
    
    # SIPaKMeD dataset uses "im_" prefix for class directories
    class_dirs = sorted([d for d in dataset_path.iterdir() 
                        if d.is_dir() and d.name.startswith("im_")])
    
    # Create class to index mapping
    class_to_idx = {cls_dir.name.replace("im_", ""): idx 
                    for idx, cls_dir in enumerate(class_dirs)}
    
    print(f"📊 Found {len(class_dirs)} classes:")
    
    for cls_dir in class_dirs:
        cls_name = cls_dir.name.replace("im_", "")
        cls_idx = class_to_idx[cls_name]
        
        # Get all image files
        images = list(cls_dir.glob("*.bmp")) + list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        
        print(f"   • {cls_name}: {len(images)} images (label: {cls_idx})")
        
        for img_path in images:
            image_paths.append(str(img_path))
            labels.append(cls_idx)
    
    print(f"\n✅ Total images loaded: {len(image_paths)}")
    
    return image_paths, labels, CLASS_NAMES


def create_splits(image_paths: List[str], labels: List[int], 
                  train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15,
                  random_state: int = SEED) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Create stratified train/val/test splits.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (paths, labels) tuples
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split sizes must sum to 1.0"
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=(val_size + test_size),
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: val vs test
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state
    )
    
    print(f"\n📊 Dataset splits:")
    print(f"   • Train: {len(X_train)} images ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"   • Val:   {len(X_val)} images ({len(X_val)/len(image_paths)*100:.1f}%)")
    print(f"   • Test:  {len(X_test)} images ({len(X_test)/len(image_paths)*100:.1f}%)")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def preprocess_image(image_path: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        label: Image label
        
    Returns:
        Tuple of (preprocessed_image, label)
    """
    # Read image file
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label


def augment_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply data augmentation to an image.
    
    Args:
        image: Input image tensor
        label: Image label
        
    Returns:
        Tuple of (augmented_image, label)
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random vertical flip
    image = tf.image.random_flip_up_down(image)
    
    # Random rotation (implemented via transpose and flip)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Random hue
    image = tf.image.random_hue(image, max_delta=0.1)
    
    # Clip values to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label


def create_dataset(image_paths: List[str], labels: List[int],
                   batch_size: int = BATCH_SIZE,
                   augment: bool = False,
                   shuffle: bool = True,
                   cache: bool = True) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset pipeline.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        batch_size: Batch size
        augment: Whether to apply data augmentation
        shuffle: Whether to shuffle the dataset
        cache: Whether to cache the dataset in memory
        
    Returns:
        tf.data.Dataset object
    """
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=SEED)
    
    # Load and preprocess images
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    
    # Cache after preprocessing
    if cache:
        dataset = dataset.cache()
    
    # Apply augmentation
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Prefetch
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset


def get_class_weights(labels: List[int]) -> Dict[int, float]:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        labels: List of all training labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # Calculate weights inversely proportional to class frequency
    weights = {int(cls): total / (len(unique) * count) 
               for cls, count in zip(unique, counts)}
    
    print(f"\n⚖️  Class weights (for handling imbalance):")
    for cls, weight in weights.items():
        print(f"   • {CLASS_NAMES[cls]}: {weight:.3f}")
    
    return weights


def save_dataset_info(splits: Dict, class_names: List[str], save_dir: str = "models"):
    """
    Save dataset information for later use.
    
    Args:
        splits: Dictionary with train/val/test splits
        class_names: List of class names
        save_dir: Directory to save the info file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    info = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'image_size': IMG_SIZE,
        'num_train': len(splits['train'][0]),
        'num_val': len(splits['val'][0]),
        'num_test': len(splits['test'][0]),
        'seed': SEED
    }
    
    info_path = save_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n💾 Dataset info saved to: {info_path}")


def main():
    """
    Main function to test the data loader.
    """
    print("=" * 60)
    print("  Data Loader Test")
    print("=" * 60)
    print()
    
    # Get dataset path
    dataset_path = get_dataset_path()
    print(f"📁 Dataset path: {dataset_path}\n")
    
    # Load image paths and labels
    image_paths, labels, class_names = load_image_paths(dataset_path)
    
    # Create splits
    splits = create_splits(image_paths, labels)
    
    # Calculate class weights
    class_weights = get_class_weights(splits['train'][1])
    
    # Create datasets
    print(f"\n🔧 Creating tf.data pipelines...")
    train_dataset = create_dataset(
        splits['train'][0], splits['train'][1],
        batch_size=BATCH_SIZE,
        augment=True,
        shuffle=True
    )
    
    val_dataset = create_dataset(
        splits['val'][0], splits['val'][1],
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )
    
    test_dataset = create_dataset(
        splits['test'][0], splits['test'][1],
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )
    
    print(f"✅ Datasets created successfully!")
    
    # Test loading a batch
    print(f"\n🧪 Testing data loading...")
    for images, labels in train_dataset.take(1):
        print(f"   • Batch shape: {images.shape}")
        print(f"   • Labels shape: {labels.shape}")
        print(f"   • Image dtype: {images.dtype}")
        print(f"   • Value range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
    
    # Save dataset info
    save_dataset_info(splits, class_names)
    
    print("\n" + "=" * 60)
    print("✅ Data loader ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
