#!/usr/bin/env python3
"""
Lip Reading System - Complete Setup and Run Script
This script will guide you through the entire process from setup to real-time prediction
"""

import os
import sys
import subprocess
from pathlib import Path
import time

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('dlib', 'dlib'),
        ('tensorflow', 'tensorflow'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('scikit-image', 'skimage'),
        ('pillow', 'PIL'),
        ('moviepy', 'moviepy'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('seaborn', 'seaborn')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_model_files():
    """Check if required model files exist"""
    print("\n🔍 Checking model files...")
    
    required_files = [
        "model/shape_predictor_68_face_landmarks.dat",
        "model/lip_reader_3dcnn.h5",
        "model/labels.npy"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    return missing_files

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = ["data", "processed_data", "model"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/")

def collect_training_data():
    """Guide user through data collection"""
    print("\n🎤 Data Collection Phase")
    print("=" * 50)
    
    print("You need to collect training data for the lip reading model.")
    print("This involves recording yourself saying different words.")
    
    response = input("\nDo you want to collect training data now? (y/n): ").lower()
    
    if response == 'y':
        print("\nStarting data collection...")
        print("Follow the instructions in the demo script.")
        
        try:
            subprocess.run([sys.executable, "demo_collection.py"], check=True)
            return True
        except subprocess.CalledProcessError:
            print("❌ Data collection failed or was interrupted.")
            return False
    else:
        print("Skipping data collection. Make sure you have data in the 'data/' directory.")
        return True

def preprocess_data():
    """Preprocess the collected data"""
    print("\n🔄 Data Preprocessing Phase")
    print("=" * 50)
    
    if not Path("data").exists() or not any(Path("data").iterdir()):
        print("❌ No data found in 'data/' directory.")
        print("Please collect training data first.")
        return False
    
    print("Preprocessing collected data...")
    
    try:
        subprocess.run([sys.executable, "src/preprocess_training.py"], check=True)
        print("✅ Data preprocessing completed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Data preprocessing failed.")
        return False

def train_model():
    """Train the lip reading model"""
    print("\n🏗️  Model Training Phase")
    print("=" * 50)
    
    if not Path("processed_data").exists():
        print("❌ No processed data found.")
        print("Please run data preprocessing first.")
        return False
    
    print("Training the 3D CNN model...")
    print("This may take a while depending on your data size and hardware.")
    
    try:
        subprocess.run([sys.executable, "src/model_training.py"], check=True)
        print("✅ Model training completed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Model training failed.")
        return False

def run_real_time_prediction():
    """Run real-time lip reading prediction"""
    print("\n🎯 Real-time Prediction Phase")
    print("=" * 50)
    
    missing_files = check_model_files()
    if missing_files:
        print("❌ Missing required model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("Please complete the training phase first.")
        return False
    
    print("Starting real-time lip reading prediction...")
    print("Look at the camera and speak clearly.")
    print("Press 'Q' to quit.")
    
    try:
        subprocess.run([sys.executable, "src/predict.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Real-time prediction failed.")
        return False

def main():
    """Main function to guide through the entire process"""
    print("🎬 Lip Reading System - Complete Setup and Run")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup directories
    setup_directories()
    
    # Check if we have a trained model
    missing_files = check_model_files()
    
    if not missing_files:
        print("\n✅ All required files found! You can run real-time prediction.")
        response = input("Do you want to run real-time prediction now? (y/n): ").lower()
        if response == 'y':
            run_real_time_prediction()
        return
    
    print(f"\n❌ Missing {len(missing_files)} required files.")
    print("We need to go through the training pipeline.")
    
    # Data collection
    if not collect_training_data():
        return
    
    # Preprocessing
    if not preprocess_data():
        return
    
    # Training
    if not train_model():
        return
    
    # Real-time prediction
    print("\n🎉 Training complete! Ready for real-time prediction.")
    response = input("Do you want to run real-time prediction now? (y/n): ").lower()
    if response == 'y':
        run_real_time_prediction()

if __name__ == "__main__":
    main() 