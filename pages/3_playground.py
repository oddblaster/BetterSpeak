import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(
    page_title="Playground",
    page_icon="📺",
)
st.sidebar.title("BetterSpeak")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Function to detect emotion based on facial landmarks
def detect_emotion(landmarks):
    # Detect smile by measuring lip corner distance
    left_lip = landmarks[61]
    right_lip = landmarks[291]
    lip_distance = calculate_distance(left_lip, right_lip)
    
    # Detect raised eyebrows
    left_eyebrow = landmarks[65]
    right_eyebrow = landmarks[295]
    eyebrow_distance = calculate_distance(left_eyebrow, right_eyebrow)
    
    if lip_distance > 0.12:  # Adjust threshold as needed
        return "Smiling"
    elif eyebrow_distance > 0.10:  # Adjust threshold as needed
        return "Surprised"
    else:
        return "Neutral"

cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

while True:
    ret, frm = cap.read()
    if not ret:
        st.error("Failed to capture frame from camera. Please check your camera connection.")
        break

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            draw.draw_landmarks(
                image=frm,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=draw.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            
            # Detect emotion
            emotion = detect_emotion(face_landmarks.landmark)
            
            # Draw emotion text on the frame
            cv2.putText(frm, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame in Streamlit
    frame_placeholder.image(frm, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()