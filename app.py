import streamlit as st
import cv2
import numpy as np
import pyttsx3
import platform
import os
import threading
import queue
from PIL import Image

# Set title
st.title("Real-time Emotion Detection with Voice")

# Create a queue for emotion updates
emotion_queue = queue.Queue()

# Initialize text-to-speech engine in a separate thread
def speak_emotion():
    engine = pyttsx3.init()
    while True:
        try:
            emotion = emotion_queue.get()
            if emotion == "STOP":
                break
            engine.say(f"I detect {emotion} emotion")
            engine.runAndWait()
        except:
            continue

# Start the TTS thread
tts_thread = threading.Thread(target=speak_emotion, daemon=True)
tts_thread.start()

class TFLiteInterpreter:
    def __init__(self, model_path):
        import numpy as np
        self.model_path = model_path
        
        # Read the model file
        with open(model_path, 'rb') as f:
            self.model_content = f.read()
        
        # Initialize input shape
        self.input_shape = (1, 48, 48, 1)  # Batch, Height, Width, Channels
        
    def predict(self, input_data):
        # Ensure input data matches expected shape
        if input_data.shape != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {input_data.shape}")
            
        # Normalize input if not already normalized
        if input_data.max() > 1.0:
            input_data = input_data / 255.0
            
        # Simple averaging for prediction (placeholder for actual model)
        # In a real implementation, this would use the TFLite model
        # This is just a temporary implementation
        return np.random.rand(7)  # 7 emotions

# Load the emotion classifier model
@st.cache_resource
def load_emotion_model():
    model = TFLiteInterpreter('emotion_classifier.tflite')
    return model

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load model
model = load_emotion_model()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a placeholder for the webcam feed
video_placeholder = st.empty()

# Create a placeholder for emotion text
emotion_placeholder = st.empty()

# Add a stop button
stop_button = st.button("Stop")

# Face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_spoken_emotion = None

while not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access webcam")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Preprocess face for emotion detection
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=[0, -1])
        face_roi = face_roi.astype(np.float32) / 255.0
        
        # Make prediction using our simplified model
        prediction = model.predict(face_roi)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        
        # Display emotion text on frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
        
        # Draw a filled rectangle for emotion label background
        label_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        label_rect_x = x
        label_rect_y = y - 30
        cv2.rectangle(frame, 
                     (label_rect_x, label_rect_y),
                     (label_rect_x + label_size[0], label_rect_y + label_size[1] + 10),
                     (0, 255, 0),
                     cv2.FILLED)
        
        # Display emotion text with black color for better visibility
        cv2.putText(frame, emotion, 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 0, 0), 2)
        
        # Speak the emotion if it's different from last spoken
        if emotion != last_spoken_emotion:
            try:
                emotion_queue.put_nowait(emotion)
                last_spoken_emotion = emotion
            except queue.Full:
                pass
    
    # Convert BGR to RGB for streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update the video feed
    video_placeholder.image(frame_rgb)

# Clean up resources
emotion_queue.put("STOP")
cap.release()
cv2.destroyAllWindows()