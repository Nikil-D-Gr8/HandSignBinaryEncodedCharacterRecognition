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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load the model with custom object scope
        with tf.keras.utils.custom_object_scope({}):
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Load metadata
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.labels = {int(k): v for k, v in self.metadata['labels'].items()}
        except FileNotFoundError:
            st.error(f"Metadata file not found at {metadata_path}")
            self.labels = {}
        except json.JSONDecodeError:
            st.error(f"Invalid JSON in metadata file at {metadata_path}")
            self.labels = {}

    def extract_hand_landmarks(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if not results.multi_hand_landmarks:
                return None
            
            landmarks = results.multi_hand_landmarks[0]
            points = []
            for landmark in landmarks.landmark:
                points.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(points)
        except Exception as e:
            st.error(f"Error in landmark extraction: {str(e)}")
            return None

    def predict(self, frame):
        try:
            landmarks = self.extract_hand_landmarks(frame)
            if landmarks is None:
                return None
            
            landmarks = landmarks.reshape(1, -1)
            prediction = self.model.predict(landmarks, verbose=0)
            predicted_class = np.argmax(prediction[0])
            
            return self.labels.get(predicted_class, "Unknown")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

def main():
    st.title("Hand Gesture Sentence Builder")
    
    # Initialize session state with default values
    if 'sentence' not in st.session_state:
        st.session_state.sentence = []
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = True
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    # Safety function to ensure sentence is always a valid list
    def ensure_valid_sentence():
        if not hasattr(st.session_state, 'sentence') or st.session_state.sentence is None:
            st.session_state.sentence = []
        if not isinstance(st.session_state.sentence, list):
            st.session_state.sentence = []

    # Button callbacks with safety checks
    def on_capture_click():
        ensure_valid_sentence()
        if st.session_state.last_prediction:
            st.session_state.sentence.append(st.session_state.last_prediction)
    
    def on_clear_click():
        ensure_valid_sentence()
        st.session_state.sentence = []

    try:
        # Initialize the recognizer
        model_path = "model_output/hand_gesture_model.h5"
        metadata_path = "model_output/model_metadata.json"
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return
            
        recognizer = HandGestureRecognizer(model_path, metadata_path)
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        # Create placeholders
        frame_placeholder = st.empty()
        sentence_placeholder = st.empty()
        prediction_placeholder = st.empty()
        
        # Buttons in columns with better spacing
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Create buttons with safe callbacks
        capture_button = col1.button("Capture", on_click=on_capture_click, key="capture_btn")
        clear_button = col2.button("Clear Sentence", on_click=on_clear_click, key="clear_btn")
        stop_button = col3.button("Stop Camera", key="stop_btn")

        # Camera handling
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to access webcam")
            return

        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            try:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = recognizer.hands.process(frame_rgb)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                
                # Get prediction
                prediction = recognizer.predict(frame)
                if prediction:
                    st.session_state.last_prediction = prediction
                    cv2.putText(frame, f"Detected: {prediction}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    prediction_placeholder.markdown(f"**Current Detection:** {prediction}")
                else:
                    prediction_placeholder.markdown("**Current Detection:** None")

                # Display the frame
                frame_placeholder.image(frame, channels="BGR")
                
                # Ensure valid sentence before display
                ensure_valid_sentence()
                
                # Display current sentence
                current_sentence = " ".join(str(word) for word in st.session_state.sentence) if st.session_state.sentence else "(Empty)"
                sentence_placeholder.markdown(f"**Current Sentence:** {current_sentence}")
                
                if stop_button:
                    st.session_state.camera_running = False
                    break

            except Exception as e:
                st.error(f"Error processing frame: {str(e)}")
                continue

        # Properly release resources
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    main()
