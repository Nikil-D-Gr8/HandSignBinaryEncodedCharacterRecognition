import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os

class HandGestureRecognizer:
    def __init__(self, model_path, metadata_path):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Load the model and metadata
        self.model = tf.keras.models.load_model(model_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Create reverse label mapping
        self.labels = {int(k): v for k, v in self.metadata['labels'].items()}

    def extract_hand_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        landmarks = results.multi_hand_landmarks[0]
        points = []
        for landmark in landmarks.landmark:
            points.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(points)

    def predict(self, frame):
        landmarks = self.extract_hand_landmarks(frame)
        if landmarks is None:
            return None
        
        # Reshape landmarks to match model input
        landmarks = landmarks.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(landmarks, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        return self.labels[predicted_class]

def main():
    st.title("Hand Gesture Sentence Builder")
    
    # Initialize session state for sentence
    if 'sentence' not in st.session_state:
        st.session_state.sentence = []

    # Initialize the recognizer
    recognizer = HandGestureRecognizer(
        model_path="model_output/hand_gesture_model.h5",
        metadata_path="model_output/model_metadata.json"
    )

    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()
    
    # Create a placeholder for the current sentence
    sentence_placeholder = st.empty()
    
    # Add buttons in a horizontal layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        capture_button = st.button("Capture")
    
    with col2:
        clear_button = st.button("Clear Sentence")
    
    with col3:
        stop_button = st.button("Stop Camera")

    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Get prediction
        prediction = recognizer.predict(frame)
        
        # Draw the prediction on the frame
        if prediction:
            cv2.putText(frame, f"Detected: {prediction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update the frame
        frame_placeholder.image(frame, channels="RGB")
        
        # Display current sentence
        current_sentence = " ".join(st.session_state.sentence)
        sentence_placeholder.markdown(f"**Current Sentence:** {current_sentence}")
        
        # Handle capture button
        if capture_button:
            if prediction:
                st.session_state.sentence.append(prediction)
            capture_button = False
            st.experimental_rerun()
        
        # Handle clear button
        if clear_button:
            st.session_state.sentence = []
            clear_button = False
            st.experimental_rerun()
        
        # Handle stop button
        if stop_button:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()