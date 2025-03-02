# Hand Gesture Text Input

A real-time hand gesture recognition system that converts hand gestures into text input using MediaPipe, TensorFlow, and Streamlit. The application uses your webcam to detect hand gestures and convert them into text characters or words.

## Features

- Real-time hand gesture recognition
- Live webcam feed with gesture predictions
- Text composition through hand gestures
- Simple and intuitive user interface
- Multi-threaded processing for smooth performance

## Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/Linux/MacOS

## Installation

1. Clone the repository:


2. Create a virtual environment (recommended):
```bash
python -m venv env
```

3. Activate the virtual environment:
- Windows:
  ```bash
  .\env\Scripts\activate
  ```
- Linux/MacOS:
  ```bash
  source env/bin/activate
  ```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Prepare your dataset:
   - Create a folder named `dataset` in the project root
   - Inside `dataset`, create subfolders for each gesture/character
   - Each subfolder should be named after the character it represents (e.g., "A", "B", "C")
   - Add training images to each subfolder (recommended: 50+ images per gesture)

Example dataset structure:
```
dataset/
    A/
        image1.jpg
        image2.jpg
        ...
    B/
        image1.jpg
        image2.jpg
        ...
```

2. Capture training data:
   - Use a webcam to capture hand gestures
   - Ensure good lighting and a clean background
   - Vary hand positions slightly for better generalization
   - Include different angles and distances

3. Run the training script:
```bash
python train.py
```

The training process:
- Extracts hand landmarks using MediaPipe (21 landmarks × 3 coordinates = 63 features)
- Processes all images in the dataset
- Trains a neural network with the following architecture:
  - Input layer: 63 features
  - Hidden layer 1: 128 neurons (ReLU)
  - Dropout layer 1: 30% dropout
  - Hidden layer 2: 64 neurons (ReLU)
  - Dropout layer 2: 20% dropout
  - Output layer: Softmax (number of classes)

The trained model and metadata will be saved in the `model_output` directory.

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. The application will open in your default web browser.

### Using the Application

1. Position your hand in front of the webcam
2. Make gestures corresponding to the characters you want to input
3. Click "Capture" to add the recognized gesture to the text
4. Use "Clear" to reset the text field

## Project Structure

```
├── app.py              # Main application file
├── train.py           # Model training script
├── requirements.txt   # Python dependencies
├── dataset/          # Training dataset directory
├── model_output/     # Trained model files
└── README.md         # Project documentation
```

## Technical Details

- **MediaPipe**: Used for hand landmark detection
- **TensorFlow**: Powers the gesture recognition model
- **Streamlit**: Provides the web interface
- **OpenCV**: Handles video capture and image processing
- **Multi-threading**: Ensures smooth performance of video processing and UI

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for their hand landmark detection solution
- TensorFlow team for their machine learning framework
- Streamlit team for their amazing web app framework
