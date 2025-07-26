# Lip Reading System - Complete Setup Guide

## 🎯 Overview

This guide will walk you through setting up and running the real-time lip reading system. The system uses a 3D CNN to analyze lip movements and classify spoken words in real-time.

## 🚀 Quick Start (Recommended)

### 1. Environment Setup

```bash
# Navigate to project directory
cd /Users/anshtandon/projects-fullstack/LipReadVox

# Set Python version (if using pyenv)
pyenv local 3.11.0

# Create virtual environment
python -m venv lipread_env

# Activate virtual environment
source lipread_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Demo

```bash
# Run the quick demo (creates sample data, trains model, tests prediction)
python quick_demo.py
```

This will:

- ✅ Create sample training data
- ✅ Preprocess the data
- ✅ Train the 3D CNN model
- ✅ Test the prediction system
- ✅ Generate training plots and confusion matrix

### 3. Test Real-time Prediction

```bash
# Run real-time lip reading prediction
python src/predict.py
```

**Instructions for real-time prediction:**

- Look at the camera clearly
- Speak one of the trained words: "hello", "goodbye", "yes", "no"
- The system will show predictions with confidence scores
- Press 'Q' to quit

## 📋 Complete Pipeline (Manual Steps)

If you want to run each step manually:

### Step 1: Data Collection

```bash
# Collect real training data
python demo_collection.py
```

**Data Collection Instructions:**

1. The system will guide you through recording different words
2. For each word, record 3-5 takes
3. Press 'R' to start recording, 'S' to stop, 'Q' to quit
4. Ensure good lighting and clear view of your face
5. Speak clearly and consistently

### Step 2: Data Preprocessing

```bash
# Preprocess collected data
python src/preprocess_training.py
```

This step:

- Processes raw video frames
- Applies image enhancement
- Creates training sequences
- Splits data into train/validation sets

### Step 3: Model Training

```bash
# Train the 3D CNN model
python src/model_training.py
```

This step:

- Builds a 3D CNN architecture
- Trains on processed data
- Saves the trained model
- Generates evaluation metrics

### Step 4: Real-time Prediction

```bash
# Run real-time prediction
python src/predict.py
```

## 🏗️ System Architecture

### Components

1. **Data Collection** (`src/collection.py`)

   - Records lip movements using webcam
   - Uses dlib for face and lip landmark detection
   - Saves frames in organized directory structure

2. **Preprocessing** (`src/preprocess_training.py`)

   - Image enhancement and normalization
   - Sequence creation (22 frames per word)
   - Train/validation split

3. **Model Training** (`src/model_training.py`)

   - 3D CNN with 4 convolutional blocks
   - Batch normalization and dropout
   - Adam optimizer with learning rate scheduling

4. **Real-time Prediction** (`src/predict.py`)
   - Live video processing
   - Frame buffer management
   - Multi-threaded prediction
   - Real-time visualization

### Model Architecture

```
Input: 22 frames × 80×112 pixels (grayscale)
├── Conv3D Block 1 (32 filters)
├── Conv3D Block 2 (64 filters)
├── Conv3D Block 3 (128 filters)
├── Conv3D Block 4 (256 filters)
├── Global Average Pooling
├── Dense Layer 1 (512 units)
├── Dense Layer 2 (256 units)
└── Output Layer (num_classes)
```

## 📊 Performance Metrics

The system generates several evaluation metrics:

- **Training History Plots** (`model/training_history.png`)

  - Training vs validation accuracy
  - Training vs validation loss

- **Confusion Matrix** (`model/confusion_matrix.png`)
  - Per-class prediction accuracy
  - Misclassification patterns

## 🔧 Customization

### Adding New Words

1. **Collect Data:**

   ```bash
   python demo_collection.py
   # Record new words during collection
   ```

2. **Retrain Model:**
   ```bash
   python src/preprocess_training.py
   python src/model_training.py
   ```

### Model Parameters

Edit `src/model_training.py` to modify:

- Model architecture (layers, filters)
- Training parameters (epochs, batch size)
- Learning rate and optimization
- Data augmentation

### Preprocessing Options

Edit `src/preprocess_training.py` to adjust:

- Image preprocessing steps
- Sequence length (currently 22 frames)
- Train/validation split ratio
- Data augmentation techniques

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected:**

   - Check camera permissions
   - Try different camera index in code
   - Ensure camera is not in use by other applications

2. **Poor face detection:**

   - Improve lighting conditions
   - Ensure face is clearly visible
   - Check dlib model file exists: `model/shape_predictor_68_face_landmarks.dat`

3. **Low prediction accuracy:**

   - Collect more training data (3-5 takes per word)
   - Improve recording quality (good lighting, clear speech)
   - Adjust confidence threshold in `src/predict.py`

4. **Memory issues during training:**

   - Reduce batch size in `src/model_training.py`
   - Use smaller model architecture
   - Reduce sequence length

5. **Dependency issues:**
   - Ensure Python 3.11 is used
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Check virtual environment is activated

### File Structure

```
LipReadVox/
├── src/
│   ├── collection.py          # Data collection
│   ├── preprocess_training.py # Data preprocessing
│   ├── model_training.py      # Model training
│   └── predict.py             # Real-time prediction
├── model/
│   ├── shape_predictor_68_face_landmarks.dat  # dlib model
│   ├── lip_reader_3dcnn.h5                    # Trained model
│   ├── labels.npy                             # Class labels
│   ├── training_history.png                   # Training plots
│   └── confusion_matrix.png                   # Evaluation metrics
├── data/                      # Raw training data
├── processed_data/            # Processed training data
├── demo_collection.py         # Data collection demo
├── quick_demo.py              # Quick system demo
├── run_lip_reader.py          # Complete setup script
└── requirements.txt           # Dependencies
```

## 🎯 Usage Examples

### Example 1: Quick Test

```bash
# Test the system with sample data
python quick_demo.py
python src/predict.py
```

### Example 2: Custom Training

```bash
# Collect your own data
python demo_collection.py

# Train custom model
python src/preprocess_training.py
python src/model_training.py

# Test custom model
python src/predict.py
```

### Example 3: Batch Processing

```bash
# Run complete pipeline
python run_lip_reader.py
```

## 📈 Expected Performance

With good training data:

- **Accuracy:** 70-90% depending on data quality
- **Real-time performance:** ~30 FPS on modern hardware
- **Memory usage:** ~2-4 GB during training
- **Model size:** ~16 MB trained model

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify Python 3.11 is being used
4. Check that the virtual environment is activated
5. Review the error messages for specific guidance

## 🎉 Success Indicators

You'll know the system is working correctly when:

- ✅ All dependencies install without errors
- ✅ Data collection captures frames successfully
- ✅ Preprocessing completes with training/validation splits
- ✅ Model training shows improving accuracy/loss
- ✅ Real-time prediction displays confidence scores
- ✅ Training plots and confusion matrix are generated
