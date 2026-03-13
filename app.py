import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="AI Gesture Draw", layout="wide", page_icon="✍️")

st.title("✍️ AI Real-Time Gesture Drawing")
st.markdown("Draw on your screen using simple hand gestures. Use the controls below to change colors or clear the canvas.")

# Instruction Section - CRITICAL FOR USER EXPERIENCE
with st.expander("📖 How to Use - Read Me First", expanded=True):
    col1, col2 = st.columns(2)
    col1.markdown("""
    ### Gestures:
    1.  **Draw Mode:** ☝️ Raise your **Index Finger** ONLY. The cursor will follow your fingertip.
    2.  **Selection Mode:** ✌️ Raise **Index + Middle Fingers** together. Move your hand to select colors/eraser from the top menu.
    3.  **Eraser:** Move your 'Selection' hand over the **'ERASE'** button at the top.
    """)
    col2.markdown("""
    ### Controls:
    * Use the sidebar to adjust the **Brush Thickness**.
    * Click **'Clear Canvas'** in the sidebar to reset everything.
    * Use **'Save Drawing'** to download your creation as a PNG.
    """)

# --- MEDIA PIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Tracks only one hand for stability
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- VARIABLES & CANVAS SETUP ---
# Defining colors (BGR format)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] # Blue, Green, Red, Yellow
colorIndex = 0 # Default to Blue

# Eraser settings
eraser_color = (0, 0, 0) # Black (will draw over existing lines on black canvas)
eraser_thickness = 50

# Cursor tracking variables
xp, yp = 0, 0

# UI - Sidebar Controls
st.sidebar.header("Drawing Tools")
brush_thickness = st.sidebar.slider("Brush Thickness", 5, 50, 20)
clear_button = st.sidebar.button("Clear Canvas")

# Initialize Session State for the canvas to persist between frames
if 'canvas' not in st.session_state or clear_button:
    # Setting up the canvas with colored buttons at the top
    # We create a black canvas that is 720p resolution
    st.session_state['canvas'] = np.zeros((720, 1280, 3), np.uint8)
    # Drawing color selection buttons (rectangles) on the canvas
    cv2.rectangle(st.session_state['canvas'], (40, 1), (140, 65), colors[0], -1) # Blue
    cv2.rectangle(st.session_state['canvas'], (160, 1), (260, 65), colors[1], -1) # Green
    cv2.rectangle(st.session_state['canvas'], (280, 1), (380, 65), colors[2], -1) # Red
    cv2.rectangle(st.session_state['canvas'], (400, 1), (500, 65), colors[3], -1) # Yellow
    # Eraser button (White rectangle with text)
    cv2.rectangle(st.session_state['canvas'], (520, 1), (650, 65), (255, 255, 255), -1)
    cv2.putText(st.session_state['canvas'], "ERASE", (540, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

# WebCam Input - Using Streamlit's built-in component
cam_input = st.camera_input("Draw Area")

# Processing the image when a frame is available
if cam_input:
    # Convert Streamlit image to OpenCV format
    file_bytes = np.asarray(bytearray(cam_input.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    # Flip the frame horizontally for mirror effect (natural interaction)
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Process Hand Landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 1. Get Landmark Coordinates
            # landmark 8 is Index Finger Tip, 12 is Middle Finger Tip
            idx_tip = hand_landmarks.landmark[8]
            mid_tip = hand_landmarks.landmark[12]
            
            # Convert normalized coordinates to pixel values
            cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
            mx, my = int(mid_tip.x * w), int(mid_tip.y * h)

            # 2. Check Gestures
            # Gesture 1: Index + Middle finger UP -> SELECTION MODE
            if idx_tip.y < hand_landmarks.landmark[6].y and mid_tip.y < hand_landmarks.landmark[10].y:
                xp, yp = 0, 0 # Reset previous points
                cv2.circle(frame, (cx, cy), 15, colors[colorIndex], cv2.FILLED)
                
                # Check for Color Selection in the top header
                if cy < 65:
                    if 40 < cx < 140: colorIndex = 0 # Blue
                    elif 160 < cx < 260: colorIndex = 1 # Green
                    elif 280 < cx < 380: colorIndex = 2 # Red
                    elif 400 < cx < 500: colorIndex = 3 # Yellow
                    elif 520 < cx < 650: colorIndex = -1 # ERASE MODE

            # Gesture 2: Index finger ONLY up -> DRAW MODE
            elif idx_tip.y < hand_landmarks.landmark[6].y and mid_tip.y > hand_landmarks.landmark[10].y:
                cv2.circle(frame, (cx, cy), 10, colors[colorIndex] if colorIndex != -1 else eraser_color, cv2.FILLED)
                
                # Drawing logic
                if xp == 0 and yp == 0:
                    xp, yp = cx, cy
                
                # Active Color/Eraser
                current_color = colors[colorIndex] if colorIndex != -1 else eraser_color
                current_thickness = brush_thickness if colorIndex != -1 else eraser_thickness

                # Draw a line from the previous point to the current point
                cv2.line(st.session_state['canvas'], (xp, yp), (cx, cy), current_color, current_thickness)
                xp, yp = cx, cy # Update previous points

            else:
                xp, yp = 0, 0 # Reset if no valid gesture

    # --- MERGING CANVAS & VIDEO ---
    # Convert canvas to grayscale
    img_gray = cv2.cvtColor(st.session_state['canvas'], cv2.COLOR_BGR2GRAY)
    # Binary thresholding to create a mask of the drawn lines
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    
    # Merge the live video frame and the inverted mask
    # This keeps the video feed where no lines are drawn
    img_final = cv2.bitwise_and(frame, img_inv)
    # Add the colored canvas lines on top
    img_final = cv2.bitwise_or(img_final, st.session_state['canvas'])

    # Display the final, merged image in Streamlit
    st.image(img_final, channels="BGR", use_container_width=True)

    # Save Option
    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Save Drawing"):
        filename = f"gesture_drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, st.session_state['canvas'])
        st.sidebar.success(f"Drawing saved as: {filename}")
else:
    st.info("💡 Please allow access to your camera below to start drawing.")
