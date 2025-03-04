import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
from threading import Thread, Lock
import queue
import time

class VideoThread(Thread):
    def __init__(self, recognizer):
        Thread.__init__(self)
        self.recognizer = recognizer
        self.frame_queue = queue.Queue(maxsize=2)
        self.prediction_queue = queue.Queue(maxsize=2)
        self.running = True
        self.daemon = True

    def run(self):
        video = cv2.VideoCapture(0)
        try:
            while self.running:
                ret, frame = video.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    result = self.recognizer.process_frame(frame)
                    
                    if result:
                        prediction, confidence = result
                        cv2.putText(
                            frame,
                            f"Prediction: {prediction} ({confidence:.2f})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        # Update prediction queue
                        try:
                            self.prediction_queue.put_nowait((prediction, confidence))
                        except queue.Full:
                            try:
                                self.prediction_queue.get_nowait()
                                self.prediction_queue.put_nowait((prediction, confidence))
                            except queue.Empty:
                                pass

                    # Update frame queue
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                            
                time.sleep(0.01)  # Small delay to prevent CPU overload
        finally:
            video.release()

    def stop(self):
        self.running = False

class HandGestureRecognizer:
    def __init__(self, model_path="model_output/hand_gesture_model.h5", metadata_path="model_output/model_metadata.json"):
        self.model = load_model(model_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        self.labels = {int(k): v for k, v in self.metadata['labels'].items()}
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Draw hand landmarks on the frame
        landmarks = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )
        
        # Extract and normalize coordinates relative to wrist
        wrist = landmarks.landmark[0]
        points = []
        
        for landmark in landmarks.landmark:
            # Normalize coordinates relative to wrist position
            points.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])
        
        # Make prediction
        prediction = self.model.predict(np.array([points]), verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return self.labels[predicted_class], confidence

def initialize_session_state():
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = HandGestureRecognizer()
    if 'video_thread' not in st.session_state:
        st.session_state.video_thread = VideoThread(st.session_state.recognizer)
        st.session_state.video_thread.start()
    if 'lock' not in st.session_state:
        st.session_state.lock = Lock()

def capture_text():
    try:
        prediction, _ = st.session_state.video_thread.prediction_queue.get_nowait()
        with st.session_state.lock:
            st.session_state.text += prediction
    except queue.Empty:
        pass

def clear_text():
    with st.session_state.lock:
        st.session_state.text = ""

def main():
    st.set_page_config(layout="centered")
    st.title("Hand Gesture Text Input")
    
    initialize_session_state()
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.text_area("Composed Text", st.session_state.text, height=100, key='text_display')
    
    with col2:
        st.button("Capture", on_click=capture_text, key='capture_btn')
        st.button("Clear", on_click=clear_text, key='clear_btn')
    
    # Display video feed
    frame_placeholder = st.empty()
    
    while True:
        try:
            frame = st.session_state.video_thread.frame_queue.get_nowait()
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        except queue.Empty:
            pass
        
        time.sleep(0.01)  # Small delay to prevent CPU overload

if __name__ == "__main__":
    main()
