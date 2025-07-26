# this will be a class that will import Dlib and OpenCV to detect faces in a video stream

import cv2
import dlib
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path

class LipReadingCollector:
    def __init__(self, word, output_dir="data", min_confidence=0.8):
        self.word = word.lower()
        self.output_dir = Path(output_dir)
        self.min_confidence = min_confidence
        
        # initialize face detection and landmark prediction based on model
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        
        # create session-specific directory and parameters for recording
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.word / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.frame_buffer = []
        self.is_recording = False
        self.quality_metrics = {
            'face_detected': False,
            'lip_visibility': 0.0,
            'motion_detected': False
        }
        
    def calculate_lip_visibility(self, landmarks):
        lip_points = list(range(48, 68))  # lip landmark area
        points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in lip_points])
        area = cv2.contourArea(points)
        # finally will return normalized image
        return min(1.0, area / 1000.0)
        
    def detect_motion(self, frame1, frame2):
        """Detect if there's significant motion between frames"""
        if frame1 is None or frame2 is None:
            return False
            
        diff = cv2.absdiff(frame1, frame2)
        return np.mean(diff) > 10.0
        
    def process_frame(self, frame):
        # now will go through each of the frames one at a time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            self.quality_metrics['face_detected'] = False
            return None
            
        self.quality_metrics['face_detected'] = True
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # lip coordinates and padding for boundaries
        lip_points = list(range(48, 68))
        x_coords = [landmarks.part(i).x for i in lip_points]
        y_coords = [landmarks.part(i).y for i in lip_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        # now will extract the lip region and resize it
        lip_region = frame[y_min:y_max, x_min:x_max]
        if lip_region.size == 0:
            return None
            
        lip_region = cv2.resize(lip_region, (112, 80))
        
        # now will update the quality metrics
        self.quality_metrics['lip_visibility'] = self.calculate_lip_visibility(landmarks)
        
        return lip_region
        
    def start_recording(self):
        self.is_recording = True
        self.frame_buffer = []
        print(f"\nRecording started for word: '{self.word}'")
        print("Press 'Q' to quit, 'S' to stop recording")
        
    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # now will save the frames if we have enough good quality ones
        if len(self.frame_buffer) >= 15:
            take_dir = self.session_dir / f"take_{len(list(self.session_dir.glob('take_*'))) + 1}"
            take_dir.mkdir(exist_ok=True)
            
            for i, frame in enumerate(self.frame_buffer):
                cv2.imwrite(str(take_dir / f"frame_{i:03d}.png"), frame)
                
            print(f"✅ Saved {len(self.frame_buffer)} frames to {take_dir}")
        else:
            print("❌ Not enough frames collected. Please try again.")
            
        self.frame_buffer = []
        
    def run(self):
        """Main collection loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            lip_region = self.process_frame(frame)
            
            # updates the motion now
            if prev_frame is not None:
                self.quality_metrics['motion_detected'] = self.detect_motion(
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                )
            prev_frame = frame.copy()
            
            status_text = [
                f"Face: {'✓' if self.quality_metrics['face_detected'] else '✗'}",
                f"Lips: {self.quality_metrics['lip_visibility']:.2f}",
                f"Motion: {'✓' if self.quality_metrics['motion_detected'] else '✗'}"
            ]
            
            # actual text within frmae
            for i, text in enumerate(status_text):
                cv2.putText(frame, text, (10, 30 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Lip Reading Collection", frame)
            if lip_region is not None:
                cv2.imshow("Lip Region", lip_region)
            
            # for key presses acocrdingly
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.stop_recording()
            elif key == ord('r') and not self.is_recording:
                self.start_recording()
                
            # frame buffer is gettinga ppended
            if self.is_recording and lip_region is not None:
                if (self.quality_metrics['face_detected'] and 
                    self.quality_metrics['lip_visibility'] > 0.5):
                    self.frame_buffer.append(lip_region)
                    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = LipReadingCollector("hello")
    collector.run()  