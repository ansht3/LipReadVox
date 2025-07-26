# LipReadVox

### Real-time Lip Reading System using 3D CNN

This project implements a real-time lip reading system that can detect and classify spoken words by analyzing lip movements using computer vision and deep learning techniques.

## 🎯 Features

- **Real-time lip detection** using dlib and OpenCV
- **3D CNN model** for temporal-spatial feature extraction
- **Live video processing** with face and lip landmark detection
- **Multi-word classification** with confidence scoring
- **Data collection tools** for training custom models
- **Comprehensive preprocessing pipeline** for optimal model performance

## 🏗️ Architecture

The system consists of four main components:

1. **Data Collection** (`src/collection.py`) - Records lip movements for training
2. **Preprocessing** (`src/preprocess_training.py`) - Processes collected data for training
3. **Model Training** (`src/model_training.py`) - Trains the 3D CNN model
4. **Real-time Prediction** (`src/predict.py`) - Runs live lip reading predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- Good lighting conditions
- Clear view of your face

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd LipReadVox
   ```

2. **Set up Python environment:**

   ```bash
   # Set Python version (if using pyenv)
   pyenv local 3.11.0

   # Create virtual environment
   python -m venv lipread_env
   source lipread_env/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download required model files:**

   ```bash
   # Download dlib shape predictor (required for face detection)
   curl -L -o model/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   cd model && bunzip2 shape_predictor_68_face_landmarks.dat.bz2 && cd ..
   ```

5. **Run the complete setup:**
   ```bash
   python quick_demo.py
   ```

This script will guide you through the entire process:

- ✅ Create sample training data
- ✅ Preprocess the data
- ✅ Train the 3D CNN model
- ✅ Test the prediction system
- ✅ Generate training plots and confusion matrix

6. **Test real-time prediction:**
   ```bash
   python src/predict.py
   ```

## 📋 Manual Setup (Alternative)

If you prefer to run each step manually:

### 1. Data Collection

```bash
python demo_collection.py
```

- Records lip movements for different words
- Press 'R' to start recording, 'S' to stop, 'Q' to quit
- Collect 3-5 takes per word for best results

### 2. Data Preprocessing

```bash
python src/preprocess_training.py
```

- Processes collected frames into training sequences
- Applies image enhancement and normalization
- Splits data into training and validation sets

### 3. Model Training

```bash
python src/model_training.py
```

- Trains a 3D CNN model on the processed data
- Saves the trained model and labels
- Generates training plots and evaluation metrics

### 4. Real-time Prediction

```bash
python src/predict.py
```

- Runs live lip reading prediction
- Shows real-time confidence scores
- Press 'Q' to quit

## 🎤 Data Collection Guidelines

For optimal model performance:

1. **Environment:**

   - Good lighting (avoid shadows on face)
   - Quiet background
   - Camera at eye level

2. **Recording:**

   - Say each word clearly and consistently
   - Maintain consistent distance from camera
   - Record 3-5 takes per word
   - Include natural variations in pronunciation

3. **Recommended Words:**
   - Start with simple words: "hello", "goodbye", "yes", "no", "thank"
   - Add more words as needed for your use case

## 🧠 Model Architecture

The system uses a 3D CNN with the following structure:

- **Input:** 22 frames × 80×112 pixels (grayscale)
- **4 Convolutional blocks** with 3D convolutions
- **Batch normalization** and dropout for regularization
- **Global average pooling** for temporal aggregation
- **Dense layers** for final classification

## 📊 Performance

Model performance depends on:

- Quality and quantity of training data
- Lighting conditions during recording
- Clarity of lip movements
- Hardware capabilities

Typical accuracy ranges from 70-90% with good training data.

## 🔧 Customization

### Adding New Words

1. Collect data for new words using `demo_collection.py`
2. Re-run preprocessing and training
3. The model will automatically include new classes

### Model Parameters

Edit `src/model_training.py` to modify:

- Model architecture
- Training parameters
- Data augmentation
- Loss functions

### Preprocessing Options

Edit `src/preprocess_training.py` to adjust:

- Image preprocessing steps
- Sequence length
- Data augmentation
- Train/validation split

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected:**

   - Check camera permissions
   - Try different camera index in code

2. **Poor face detection:**

   - Improve lighting conditions
   - Ensure face is clearly visible
   - Check dlib model file exists: `model/shape_predictor_68_face_landmarks.dat`

3. **Low prediction accuracy:**

   - Collect more training data
   - Improve recording quality
   - Adjust confidence threshold

4. **Memory issues during training:**

   - Reduce batch size
   - Use smaller model architecture
   - Reduce sequence length

5. **Missing model files:**
   - Download the dlib shape predictor (see DOWNLOAD_MODELS.md)
   - Run `python quick_demo.py` to generate trained model files

### File Structure

```
LipReadVox/
├── src/
│   ├── collection.py          # Data collection
│   ├── preprocess_training.py # Data preprocessing
│   ├── model_training.py      # Model training
│   └── predict.py             # Real-time prediction
├── model/
│   ├── shape_predictor_68_face_landmarks.dat  # Downloaded (95MB)
│   ├── lip_reader_3dcnn.h5                    # Generated by training (~16MB)
│   └── labels.npy                             # Generated by training (~240B)
├── data/                      # Raw training data (not in repo)
├── processed_data/            # Processed training data (not in repo)
├── demo_collection.py         # Data collection demo
├── quick_demo.py              # Quick system demo
├── run_lip_reader.py          # Complete setup script
├── DOWNLOAD_MODELS.md         # Model download instructions
└── requirements.txt           # Dependencies
```

### Next Steps

1. **Improved model generalization with a diverse dataset.**
2. **Real world deployment / integration within vision system**
