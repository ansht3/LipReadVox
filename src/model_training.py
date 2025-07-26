import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class LipReadingModel:
    def __init__(self, input_shape=(22, 80, 112, 1), num_classes=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_3d_cnn(self):
        """Build a 3D CNN model for lip reading"""
        model = tf.keras.Sequential([
            # First 3D Convolutional Block
            tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', 
                                  input_shape=self.input_shape, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second 3D Convolutional Block
            tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third 3D Convolutional Block
            tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth 3D Convolutional Block
            tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Global Average Pooling
            tf.keras.layers.GlobalAveragePooling3D(),
            
            # Dense Layers
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            # Output Layer
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ['accuracy']
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def train(self, train_dataset, val_dataset, epochs=50, callbacks=None):
        """Train the model"""
        if callbacks is None:
            callbacks = self.get_default_callbacks()
            
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def get_default_callbacks(self):
        """Get default training callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'model/lip_reader_3dcnn.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def evaluate(self, test_dataset, labels):
        """Evaluate the model and generate reports"""
        # Get predictions
        predictions = self.model.predict(test_dataset)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_labels = []
        for batch_x, batch_y in test_dataset:
            true_labels.extend(batch_y.numpy())
        true_labels = np.array(true_labels)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_classes, target_names=labels))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(true_labels, predicted_classes, labels)
        
        return predictions, predicted_classes
    
    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('model/confusion_matrix.png')
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('model/training_history.png')
        plt.show()

def main():
    """Main training function"""
    # Load processed data
    processed_dir = Path("processed_data")
    
    if not processed_dir.exists():
        print("❌ Processed data not found. Please run preprocess_training.py first.")
        return
    
    # Load data
    X_train = np.load(processed_dir / "train" / "X_train.npy")
    y_train = np.load(processed_dir / "train" / "y_train.npy")
    X_val = np.load(processed_dir / "val" / "X_val.npy")
    y_val = np.load(processed_dir / "val" / "y_val.npy")
    
    # Load labels
    labels = np.load(processed_dir / "label_encoder.npy", allow_pickle=True)
    num_classes = len(labels)
    
    print(f"📊 Dataset loaded:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {list(labels)}")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # Configure datasets
    batch_size = 16  # Smaller batch size for 3D CNN
    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Build and train model
    print("\n🏗️  Building 3D CNN model...")
    model_trainer = LipReadingModel(num_classes=num_classes)
    model_trainer.model = model_trainer.build_3d_cnn()
    model_trainer.compile_model()
    
    # Print model summary
    model_trainer.model.summary()
    
    # Train the model
    print("\n🚀 Starting training...")
    history = model_trainer.train(train_dataset, val_dataset, epochs=50)
    
    # Save labels for prediction
    np.save("model/labels.npy", labels)
    
    # Plot training history
    model_trainer.plot_training_history(history)
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    model_trainer.evaluate(val_dataset, labels)
    
    print("\n✅ Training complete! Model saved to model/lip_reader_3dcnn.h5")
    print("Labels saved to model/labels.npy")

if __name__ == "__main__":
    main()
