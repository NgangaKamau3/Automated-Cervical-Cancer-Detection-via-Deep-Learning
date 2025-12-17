"""
Dataset Download Script
Downloads the SIPaKMeD cervical cancer dataset from Kaggle using kagglehub.
"""

import os
import sys
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("❌ kagglehub not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

def download_sipakmed_dataset(target_dir="data/raw"):
    """
    Download the SIPaKMeD cervical cancer dataset from Kaggle.
    
    Args:
        target_dir: Directory where the dataset will be stored
        
    Returns:
        Path to the downloaded dataset
    """
    print("🔽 Downloading SIPaKMeD dataset from Kaggle...")
    print("   This may take a few minutes depending on your connection.\n")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed")
        
        print(f"✅ Dataset downloaded successfully!")
        print(f"📁 Path to dataset files: {path}\n")
        
        # Create target directory if it doesn't exist
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Create a symlink or copy the data to the project directory
        source_path = Path(path)
        
        # Check if data was downloaded properly
        if not source_path.exists():
            raise FileNotFoundError(f"Downloaded path does not exist: {path}")
        
        # List contents of downloaded directory
        print("📊 Dataset contents:")
        for item in source_path.iterdir():
            if item.is_dir():
                file_count = len(list(item.rglob("*")))
                print(f"   📂 {item.name}/ ({file_count} files)")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"   📄 {item.name} ({size_mb:.2f} MB)")
        
        print(f"\n💡 Dataset cached at: {path}")
        print(f"💡 You can access it directly from this location in your training scripts.")
        
        # Save the path to a config file for easy access
        config_file = Path("config/dataset_path.txt")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(str(path))
        print(f"💾 Dataset path saved to: {config_file}")
        
        return str(path)
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you have a Kaggle account")
        print("   2. Ensure your Kaggle API credentials are configured")
        print("      (https://www.kaggle.com/docs/api)")
        print("   3. Check your internet connection")
        raise

def verify_dataset_structure(dataset_path):
    """
    Verify that the dataset has the expected structure.
    
    Args:
        dataset_path: Path to the downloaded dataset
    """
    dataset_path = Path(dataset_path)
    
    print("\n🔍 Verifying dataset structure...")
    
    # Expected classes for SIPaKMeD dataset
    expected_classes = [
        "im_Dyskeratotic",
        "im_Koilocytotic", 
        "im_Metaplastic",
        "im_Parabasal",
        "im_Superficial-Intermediate"
    ]
    
    found_classes = []
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir() and class_dir.name.startswith("im_"):
            found_classes.append(class_dir.name)
            
            # Count images in this class recursively
            image_files = list(class_dir.rglob("*.bmp")) + list(class_dir.rglob("*.jpg")) + list(class_dir.rglob("*.png"))
            print(f"   ✓ {class_dir.name}: {len(image_files)} images")
    
    # Check if all expected classes are present
    missing_classes = set(expected_classes) - set(found_classes)
    if missing_classes:
        print(f"\n⚠️  Warning: Missing expected classes: {missing_classes}")
    else:
        print(f"\n✅ All {len(expected_classes)} expected classes found!")
    
    return found_classes

def main():
    """Main execution function."""
    print("=" * 60)
    print("  SIPaKMeD Cervical Cancer Dataset Downloader")
    print("=" * 60)
    print()
    
    try:
        # Download the dataset
        dataset_path = download_sipakmed_dataset()
        
        # Verify structure
        classes = verify_dataset_structure(dataset_path)
        
        print("\n" + "=" * 60)
        print("✅ Dataset ready for training!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Run data preprocessing: python ml/data_loader.py")
        print(f"  2. Start training: python ml/train.py")
        print()
        
        return dataset_path
        
    except Exception as e:
        print(f"\n❌ Failed to download dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
