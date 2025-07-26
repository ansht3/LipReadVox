import cv2
import numpy as np
import tensorflow as tf
import dlib
from pathlib import Path
import time
from collections import deque
import threading
import queue

class LipReaderPredictor:
    def __init__(self, model_path="model/lip_reader_3dcnn.h5", label_path="model/labels.npy"):
        # Check if model and labels exist
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first using model_training.py")
            return
            
        if not Path(label_path).exists():
            print(f"❌ Labels not found at {label_path}")
            print("Please train the model first using model_training.py")
            return
            
        self.model = tf.keras.models.load_model(model_path)
        self.labels = np.load(label_path, allow_pickle=True)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        
        # random buffer for frames
        self.frame_buffer = deque(maxlen=22)
        self.processing_queue = queue.Queue()
        self.prediction_thread = None
        self.is_running = False
        
        # visualization parameters
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.7
        
        print(f"✅ Model loaded successfully!")
        print(f"   Classes: {list(self.labels)}")
        print(f"   Model path: {model_path}")
        
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        # gaussian blur and normalize
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
        enhanced = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        normalized = enhanced.astype(np.float32) / 255.0
        return normalized
        
    def extract_lip_region(self, frame, landmarks):
        # get lip landmarks
        lip_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 68)])
        
        # calculate bounding box with margin
        margin = 0.3
        x_min, y_min = np.min(lip_points, axis=0)
        x_max, y_max = np.max(lip_points, axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, int(x_min - width * margin))
        x_max = min(frame.shape[1], int(x_max + width * margin))
        y_min = max(0, int(y_min - height * margin))
        y_max = min(frame.shape[0], int(y_max + height * margin))
        lip_region = frame[y_min:y_max, x_min:x_max]
        return cv2.resize(lip_region, (112, 80))
        
    def prediction_worker(self):
        while self.is_running:
            try:
                frames = self.processing_queue.get(timeout=1)
                if frames is None:
                    continue
                    
                # prepare input
                input_sequence = np.array(frames)
                input_sequence = np.expand_dims(input_sequence, axis=0)
                input_sequence = np.expand_dims(input_sequence, axis=-1)
                
                # get prediction
                predictions = self.model.predict(input_sequence, verbose=0)
                predicted_idx = np.argmax(predictions)
                confidence = predictions[0][predicted_idx]
                
                if confidence > self.confidence_threshold:
                    predicted_word = self.labels[predicted_idx]
                    self.prediction_history.append((predicted_word, confidence))
                    
            except queue.Empty:
                continue
                
    def process_frame(self, frame):
        # face detection and landmark detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return frame
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # extract and preprocess lip region
        lip_region = self.extract_lip_region(frame, landmarks)
        processed_frame = self.preprocess_frame(lip_region)
        self.frame_buffer.append(processed_frame)
        
        # add to processing queue
        if len(self.frame_buffer) == self.frame_buffer.maxlen:
            self.processing_queue.put(list(self.frame_buffer))
            
        # draw visualization
        self.draw_visualization(frame, landmarks)
        
        return frame
        
    def draw_visualization(self, frame, landmarks):
        # draw lip landmarks
        for i in range(48, 68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        y_offset = 30
        for word, conf in self.prediction_history:
            text = f"{word}: {conf:.2f}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
    def start(self):
        self.is_running = True
        self.prediction_thread = threading.Thread(target=self.prediction_worker)
        self.prediction_thread.start()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        print("\nLip Reading System Started")
        print("Press 'Q' to quit")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # processing the frame
            processed_frame = self.process_frame(frame)
            cv2.imshow("Lip Reader", processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.is_running = False
        if self.prediction_thread:
            self.prediction_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        
def main():
    predictor = LipReaderPredictor()
    predictor.start()

if __name__ == "__main__":
    main()
