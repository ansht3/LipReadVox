import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

class LipReadingPreprocessor:
    def __init__(self, data_dir="data", output_dir="processed_data", sequence_length=22):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.target_size = (112, 80)  # (width, height)
        
        # create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        
        # setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def apply_preprocessing(self, image):
        """Apply all preprocessing steps to a single image"""
        # convert to grayscale if not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # apply gaussian blur + contrast stretching
        image = cv2.GaussianBlur(image, (5, 5), 0)
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip(image, p2, p98)
        image = ((image - p2) / (p98 - p2) * 255).astype(np.uint8)
        
        # bilateral filtering
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # edge sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        
        # normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
        
    def process_sequence(self, sequence_path):
        # process a sequence of frames
        frames = []
        frame_files = sorted(sequence_path.glob("frame_*.png"))
        
        # ensure we have enough frames
        if len(frame_files) < self.sequence_length:
            self.logger.warning(f"Not enough frames in {sequence_path}")
            return None
            
        # process each frame
        for frame_file in frame_files[:self.sequence_length]:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                self.logger.error(f"Could not read frame: {frame_file}")
                return None
                
            # resize and preprocess
            frame = cv2.resize(frame, self.target_size)
            frame = self.apply_preprocessing(frame)
            frames.append(frame)
            
        return np.array(frames)
        
    def prepare_dataset(self): #preparing dataset to be sent into trainingin for model
        sequences = []
        labels = []
        for word_dir in tqdm(list(self.data_dir.glob("*")), desc="Processing words"):
            if not word_dir.is_dir():
                continue
                
            word = word_dir.name
            for session_dir in word_dir.glob("*"):
                if not session_dir.is_dir():
                    continue
                    
                for take_dir in session_dir.glob("take_*"):
                    sequence = self.process_sequence(take_dir)
                    if sequence is not None:
                        sequences.append(sequence)
                        labels.append(word)
        
        if not sequences:
            self.logger.error("No valid sequences found!")
            return
            
        # encoding and spliting into train and test accordingly
        X = np.array(sequences)
        y = np.array(labels)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        np.save(self.output_dir / "label_encoder.npy", label_encoder.classes_)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        np.save(self.output_dir / "train" / "X_train.npy", X_train)
        np.save(self.output_dir / "train" / "y_train.npy", y_train)
        np.save(self.output_dir / "val" / "X_val.npy", X_val)
        np.save(self.output_dir / "val" / "y_val.npy", y_val)
        
        self.logger.info(f"Processed {len(sequences)} sequences")
        self.logger.info(f"Training set shape: {X_train.shape}")
        self.logger.info(f"Validation set shape: {X_val.shape}")
        
    def create_tf_dataset(self, batch_size=32):
        """Create TensorFlow datasets for training"""
        # Load processed data
        X_train = np.load(self.output_dir / "train" / "X_train.npy")
        y_train = np.load(self.output_dir / "train" / "y_train.npy")
        X_val = np.load(self.output_dir / "val" / "X_val.npy")
        y_val = np.load(self.output_dir / "val" / "y_val.npy")
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        
        # Configure datasets
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset

def main():
    # initialize preprocessor and creating tensorflow datasets
    preprocessor = LipReadingPreprocessor()
    preprocessor.prepare_dataset()
    train_dataset, val_dataset = preprocessor.create_tf_dataset()
    
    print("\nDataset preparation complete!")
    print("Training dataset ready for model training.")
    print("Use the following code to load the model:")
    print("""
    model = tf.keras.models.load_model('model/lip_reader_3dcnn.h5')
    model.fit(train_dataset, validation_data=val_dataset, epochs=50)
    """)

if __name__ == "__main__":
    main()
