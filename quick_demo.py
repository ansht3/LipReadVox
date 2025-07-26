#!/usr/bin/env python3
"""
Quick Demo for Lip Reading System
This script demonstrates the system working with sample data
"""

import sys
import numpy as np
from pathlib import Path
sys.path.append('src')

def create_sample_data():
    """Create sample training data for demonstration"""
    print("🎬 Creating sample training data...")
    
    # Create sample data structure
    words = ["hello", "goodbye", "yes", "no"]
    
    for word in words:
        word_dir = Path(f"data/{word}")
        word_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 2 sessions per word
        for session in range(2):
            session_dir = word_dir / f"session_{session}"
            session_dir.mkdir(exist_ok=True)
            
            # Create 2 takes per session
            for take in range(2):
                take_dir = session_dir / f"take_{take + 1}"
                take_dir.mkdir(exist_ok=True)
                
                # Create 22 sample frames (112x80 grayscale)
                for i in range(22):
                    # Create random lip-like images
                    frame = np.random.randint(0, 255, (80, 112), dtype=np.uint8)
                    # Add some structure to make it look more like lips
                    frame[30:50, 40:72] = np.random.randint(100, 200, (20, 32))
                    
                    frame_path = take_dir / f"frame_{i:03d}.png"
                    import cv2
                    cv2.imwrite(str(frame_path), frame)
    
    print(f"✅ Created sample data for {len(words)} words")
    return words

def run_preprocessing():
    """Run the preprocessing step"""
    print("\n🔄 Running data preprocessing...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "src/preprocess_training.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Data preprocessing completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_training():
    """Run the model training"""
    print("\n🏗️  Running model training...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "src/model_training.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Model training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_prediction():
    """Test the prediction system"""
    print("\n🎯 Testing prediction system...")
    
    # Check if model files exist
    model_path = Path("model/lip_reader_3dcnn.h5")
    labels_path = Path("model/labels.npy")
    
    if not model_path.exists() or not labels_path.exists():
        print("❌ Model files not found. Please run training first.")
        return False
    
    print("✅ Model files found!")
    print("You can now run real-time prediction with:")
    print("python src/predict.py")
    
    return True

def main():
    """Main demo function"""
    print("🎬 Lip Reading System - Quick Demo")
    print("=" * 50)
    
    # Step 1: Create sample data
    words = create_sample_data()
    
    # Step 2: Run preprocessing
    if not run_preprocessing():
        print("❌ Demo failed at preprocessing step")
        return
    
    # Step 3: Run training
    if not run_training():
        print("❌ Demo failed at training step")
        return
    
    # Step 4: Test prediction
    if not test_prediction():
        print("❌ Demo failed at prediction step")
        return
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run real-time prediction: python src/predict.py")
    print("2. Collect real training data: python demo_collection.py")
    print("3. Retrain with real data: python src/model_training.py")

if __name__ == "__main__":
    main() 