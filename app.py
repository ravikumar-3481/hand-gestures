import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="AI Gesture Draw", layout="wide", page_icon="✍️")

st.title("✍️ AI Real-Time Gesture Drawing")
st.markdown("Draw on your screen using simple hand gestures. Optimized for Streamlit Cloud.")

# --- ROBUST MEDIAPIPE INITIALIZATION ---
# Using session state to ensure hands model is initialized only once
if 'hands_model' not in st.session_state:
    st.session_state.mp_hands = mp.solutions.hands
    st.session_state.hands_model = st.session_state.mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    st.session_state.mp_draw = mp.solutions.drawing_utils

# --- VARIABLES & CANVAS SETUP ---
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] # BGR: Blue, Green, Red, Yellow
if 'colorIndex' not in st.session_state:
    st.session_state.colorIndex = 0

# Sidebar Controls
st.sidebar.header("Drawing Tools")
brush_thickness = st.sidebar.slider("Brush Thickness", 5, 50, 20)
clear_button = st.sidebar.button("Clear Canvas")

# Persistent Canvas Setup
if 'canvas' not in st.session_state or clear_button:
    st.session_state.canvas = np.zeros((720, 1280, 3), np.uint8)
    # Header UI on Canvas
    cv2.rectangle(st.session_state.canvas, (40, 1), (140, 80), colors[0], -1) 
    cv2.rectangle(st.session_state.canvas, (160, 1), (260, 80), colors[1], -1) 
    cv2.rectangle(st.session_state.canvas, (280, 1), (380, 80), colors[2], -1) 
    cv2.rectangle(st.session_state.canvas, (400, 1), (500, 80), colors[3], -1) 
    cv2.rectangle(st.session_state.canvas, (520, 1), (650, 80), (255, 255, 255), -1)
    cv2.putText(st.session_state.canvas, "ERASE", (545, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Global tracking for points
if 'xp' not in st.session_state:
    st.session_state.xp, st.session_state.yp = 0, 0

# Camera Input
cam_input = st.camera_input("Draw Area (Stay in frame)")

if cam_input:
    # Convert Streamlit image to OpenCV format
    file_bytes = np.asarray(bytearray(cam_input.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Process Hand
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = st.session_state.hands_model.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get Index and Middle finger tips
            idx_tip = hand_landmarks.landmark[8]
            mid_tip = hand_landmarks.landmark[12]
            
            cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
            mx, my = int(mid_tip.x * w), int(mid_tip.y * h)

            # Selection Mode: Two fingers UP
            if idx_tip.y < hand_landmarks.landmark[6].y and mid_tip.y < hand_landmarks.landmark[10].y:
                st.session_state.xp, st.session_state.yp = 0, 0
                cv2.circle(frame, (cx, cy), 15, (255, 255, 255), cv2.FILLED)
                
                if cy < 80:
                    if 40 < cx < 140: st.session_state.colorIndex = 0
                    elif 160 < cx < 260: st.session_state.colorIndex = 1
                    elif 280 < cx < 380: st.session_state.colorIndex = 2
                    elif 400 < cx < 500: st.session_state.colorIndex = 3
                    elif 520 < cx < 650: st.session_state.colorIndex = -1 # Eraser

            # Draw Mode: Index finger ONLY up
            elif idx_tip.y < hand_landmarks.landmark[6].y and mid_tip.y > hand_landmarks.landmark[10].y:
                draw_color = (0, 0, 0) if st.session_state.colorIndex == -1 else colors[st.session_state.colorIndex]
                thickness = 50 if st.session_state.colorIndex == -1 else brush_thickness
                
                cv2.circle(frame, (cx, cy), 10, draw_color, cv2.FILLED)
                
                if st.session_state.xp == 0 and st.session_state.yp == 0:
                    st.session_state.xp, st.session_state.yp = cx, cy

                cv2.line(st.session_state.canvas, (st.session_state.xp, st.session_state.yp), (cx, cy), draw_color, thickness)
                st.session_state.xp, st.session_state.yp = cx, cy
            else:
                st.session_state.xp, st.session_state.yp = 0, 0

    # Merging Logic
    img_gray = cv2.cvtColor(st.session_state.canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img_final = cv2.bitwise_and(frame, img_inv)
    img_final = cv2.bitwise_or(img_final, st.session_state.canvas)

    st.image(img_final, channels="BGR", use_container_width=True)

    if st.sidebar.button("💾 Save Drawing"):
        filename = f"drawing_{datetime.now().strftime('%H%M%S')}.png"
        cv2.imwrite(filename, st.session_state.canvas)
        st.sidebar.success("Saved!")
