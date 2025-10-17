import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

def initialize_camera():
    """Try different camera indices to find a working camera"""
    for camera_index in range(2):  # Try first two camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return cap, camera_index
            cap.release()
    return None, None

# Set page config for a cleaner look
st.set_page_config(
    page_title="Emotion Detector",
    layout="wide"
)

# Set title
st.title("Real-time Emotion Detection")

# Sidebar controls
st.sidebar.header("Controls")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 2)

# Initialize face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Error loading face detection model: {str(e)}")
    st.stop()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize video capture
cap, camera_index = initialize_camera()
if cap is None:
    st.error("No working camera found. Please check your camera connections.")
    st.stop()

# Create placeholders for video and status
video_placeholder = st.empty()
status_placeholder = st.sidebar.empty()
emotion_placeholder = st.sidebar.empty()

# Add camera info to sidebar
st.sidebar.info(f"Using camera index: {camera_index}")

# Function to process frame
def process_frame(frame, frame_count):
    if frame_count % frame_skip != 0:
        return frame, None
    
    # Convert to grayscale
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        st.error(f"Error converting frame to grayscale: {str(e)}")
        return frame, None

    # Detect faces
    try:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
    except Exception as e:
        st.error(f"Error detecting faces: {str(e)}")
        return frame, None

    current_emotion = None
    
    for (x, y, w, h) in faces:
        try:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract and preprocess face
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = np.expand_dims(face_roi, axis=[0, -1])
            face_roi = face_roi.astype(np.float32) / 255.0
            
            # Get random emotion for now (replace with actual model prediction)
            # This is just for testing the UI
            current_emotion = np.random.choice(emotion_labels)
            
            # Draw emotion label
            label_size = cv2.getTextSize(current_emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, 
                         (x, y - 30), 
                         (x + label_size[0], y), 
                         (0, 255, 0), 
                         cv2.FILLED)
            cv2.putText(frame, current_emotion, 
                       (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 0), 2)
        except Exception as e:
            st.error(f"Error processing face: {str(e)}")
            continue
    
    return frame, current_emotion

# Main loop
frame_count = 0
stop_button = st.sidebar.button("Stop")

while not stop_button:
    try:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            break

        # Process frame
        frame_count += 1
        processed_frame, current_emotion = process_frame(frame, frame_count)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Update video feed
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # Update status
        status_placeholder.text("Status: Running")
        if current_emotion:
            emotion_placeholder.info(f"Detected Emotion: {current_emotion}")
        
        # Control frame rate
        time.sleep(0.03)  # ~30 FPS
        
    except Exception as e:
        st.error(f"Error in main loop: {str(e)}")
        break

# Add stop button outside the loop
if st.sidebar.button("Exit Application"):
    # Clean up
    try:
        cap.release()
        cv2.destroyAllWindows()
        status_placeholder.text("Status: Stopped")
        st.stop()
    except Exception as e:
        st.error(f"Error cleaning up: {str(e)}")
        st.stop()