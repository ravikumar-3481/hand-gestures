import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="AI Gesture Draw", layout="wide")

st.title("✍️ AI Gesture Drawing Board")
st.write("Raise your index finger to draw. Use index + middle fingers to select colors or erase.")

# --- STATIC MEDIAPIPE INITIALIZATION ---
# This is the most stable way to initialize MediaPipe on Streamlit Cloud
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def get_hands_model():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

hands = get_hands_model()

# --- CANVAS SETTINGS ---
# Define colors in BGR format
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] # Blue, Green, Red, Yellow
color_labels = ["Blue", "Green", "Red", "Yellow", "Eraser"]

if 'color_idx' not in st.session_state:
    st.session_state.color_idx = 0

# Sidebar controls
st.sidebar.header("Controls")
brush_size = st.sidebar.slider("Brush Thickness", 5, 50, 20)
if st.sidebar.button("Clear Canvas"):
    st.session_state.canvas = np.zeros((720, 1280, 3), np.uint8)
    st.rerun()

# --- CANVAS INITIALIZATION ---
if 'canvas' not in st.session_state:
    st.session_state.canvas = np.zeros((720, 1280, 3), np.uint8)

# Previous tracking points
if 'prev_x' not in st.session_state:
    st.session_state.prev_x, st.session_state.prev_y = 0, 0

# --- CAMERA INPUT ---
img_file = st.camera_input("Position yourself in front of the camera")

if img_file:
    # Convert Streamlit image to OpenCV format
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame = cv2.flip(frame, 1) # Mirror effect
    h, w, c = frame.shape

    # MediaPipe Processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # UI Header on Canvas (Selections)
    cv2.rectangle(st.session_state.canvas, (10, 10), (150, 80), colors[0], -1)
    cv2.rectangle(st.session_state.canvas, (170, 10), (310, 80), colors[1], -1)
    cv2.rectangle(st.session_state.canvas, (330, 10), (470, 80), colors[2], -1)
    cv2.rectangle(st.session_state.canvas, (490, 10), (630, 80), colors[3], -1)
    cv2.rectangle(st.session_state.canvas, (650, 10), (800, 80), (255, 255, 255), -1)
    cv2.putText(st.session_state.canvas, "ERASER", (675, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Tip of index and middle fingers
            # Landmark 8 = Index Tip, 12 = Middle Tip
            idx_x = int(hand_lms.landmark[8].x * w)
            idx_y = int(hand_lms.landmark[8].y * h)
            mid_x = int(hand_lms.landmark[12].x * w)
            mid_y = int(hand_lms.landmark[12].y * h)

            # Check if fingers are up (Selection Mode = 2 fingers)
            if hand_lms.landmark[8].y < hand_lms.landmark[6].y and hand_lms.landmark[12].y < hand_lms.landmark[10].y:
                st.session_state.prev_x, st.session_state.prev_y = 0, 0 # Reset drawing
                
                # Selection Logic
                if idx_y < 80:
                    if 10 < idx_x < 150: st.session_state.color_idx = 0
                    elif 170 < idx_x < 310: st.session_state.color_idx = 1
                    elif 330 < idx_x < 470: st.session_state.color_idx = 2
                    elif 490 < idx_x < 630: st.session_state.color_idx = 3
                    elif 650 < idx_x < 800: st.session_state.color_idx = -1 # Eraser
                
                cv2.circle(frame, (idx_x, idx_y), 20, (255, 255, 255), cv2.FILLED)

            # Drawing Mode (Only index finger up)
            elif hand_lms.landmark[8].y < hand_lms.landmark[6].y:
                cv2.circle(frame, (idx_x, idx_y), 15, colors[st.session_state.color_idx] if st.session_state.color_idx != -1 else (0,0,0), cv2.FILLED)
                
                if st.session_state.prev_x == 0 and st.session_state.prev_y == 0:
                    st.session_state.prev_x, st.session_state.prev_y = idx_x, idx_y

                draw_color = colors[st.session_state.color_idx] if st.session_state.color_idx != -1 else (0,0,0)
                thickness = brush_size if st.session_state.color_idx != -1 else 60

                cv2.line(st.session_state.canvas, (st.session_state.prev_x, st.session_state.prev_y), (idx_x, idx_y), draw_color, thickness)
                st.session_state.prev_x, st.session_state.prev_y = idx_x, idx_y
            else:
                st.session_state.prev_x, st.session_state.prev_y = 0, 0

    # Overlay Canvas on Video
    gray_canvas = cv2.cvtColor(st.session_state.canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    
    combined_img = cv2.bitwise_and(frame, inv_mask)
    combined_img = cv2.bitwise_or(combined_img, st.session_state.canvas)

    st.image(combined_img, channels="BGR", use_container_width=True)
