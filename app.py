import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="AI Virtual Gesture Painter", layout="wide")

class GesturePainter:
    def __init__(self):
        # Initialize MediaPipe Hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Canvas properties
        self.draw_color = (255, 0, 255) # Default: Purple
        self.brush_thickness = 15
        self.eraser_thickness = 100
        self.xp, self.yp = 0, 0 # Previous coordinates
        
        # Colors (BGR)
        self.colors = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "Eraser": (0, 0, 0)
        }
        
        # Internal State
        self.canvas = None

    def get_finger_status(self, hand_landmarks):
        """Identifies which fingers are up."""
        # MediaPipe finger tip IDs
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        
        # Thumb (Simple horizontal check for right hand)
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # 4 Fingers
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def draw_header(self, img):
        """Draws the selection UI at the top of the frame."""
        h, w, _ = img.shape
        # Header background
        cv2.rectangle(img, (0, 0), (w, 100), (50, 50, 50), cv2.FILLED)
        
        # Color boxes
        cv2.rectangle(img, (10, 10), (150, 90), (0, 0, 255), cv2.FILLED) # Red
        cv2.putText(img, "RED", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(img, (160, 10), (300, 90), (0, 255, 0), cv2.FILLED) # Green
        cv2.putText(img, "GREEN", (185, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(img, (310, 10), (450, 90), (255, 0, 0), cv2.FILLED) # Blue
        cv2.putText(img, "BLUE", (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(img, (460, 10), (600, 90), (200, 200, 200), cv2.FILLED) # Eraser
        cv2.putText(img, "ERASER", (485, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return img

    def run(self):
        st.title("🎨 AI Virtual Gesture Painter")
        st.markdown("""
        ### How to use:
        1.  **Selection Mode:** Raise **Index + Middle** fingers to pick a color from the top bar.
        2.  **Drawing Mode:** Raise **only the Index** finger to draw.
        3.  **Reset:** Use the button in the sidebar to clear the canvas.
        """)
        
        run_app = st.sidebar.checkbox('Start Webcam', value=True)
        if st.sidebar.button('Clear Canvas'):
            self.canvas = None
            st.success("Canvas Cleared!")

        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)

        while run_app:
            success, frame = cap.read()
            if not success:
                st.error("Could not access webcam.")
                break

            # Pre-process frame
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Initialize canvas if it doesn't exist
            if self.canvas is None:
                self.canvas = np.zeros((h, w, 3), np.uint8)

            # Mediapipe detection
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            frame = self.draw_header(frame)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Get index and middle finger tip coordinates
                    # Landmark 8: Index Tip, Landmark 12: Middle Tip
                    lm8 = hand_lms.landmark[8]
                    lm12 = hand_lms.landmark[12]
                    x1, y1 = int(lm8.x * w), int(lm8.y * h)
                    x2, y2 = int(lm12.x * w), int(lm12.y * h)

                    fingers = self.get_finger_status(hand_lms)

                    # 1. SELECTION MODE (Index & Middle fingers are up)
                    if fingers[1] and fingers[2]:
                        self.xp, self.yp = 0, 0
                        cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), self.draw_color, cv2.FILLED)
                        
                        # Logic for choosing color based on header coordinates
                        if y1 < 100:
                            if 10 < x1 < 150:
                                self.draw_color = self.colors["Red"]
                            elif 160 < x1 < 300:
                                self.draw_color = self.colors["Green"]
                            elif 310 < x1 < 450:
                                self.draw_color = self.colors["Blue"]
                            elif 460 < x1 < 600:
                                self.draw_color = self.colors["Eraser"]

                    # 2. DRAWING MODE (Only Index finger is up)
                    elif fingers[1] and not fingers[2]:
                        cv2.circle(frame, (x1, y1), 15, self.draw_color, cv2.FILLED)
                        
                        if self.xp == 0 and self.yp == 0:
                            self.xp, self.yp = x1, y1
                        
                        if self.draw_color == (0, 0, 0): # Eraser logic
                            cv2.line(frame, (self.xp, self.yp), (x1, y1), (0, 0, 0), self.eraser_thickness)
                            cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), (0, 0, 0), self.eraser_thickness)
                        else:
                            cv2.line(frame, (self.xp, self.yp), (x1, y1), self.draw_color, self.brush_thickness)
                            cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.brush_thickness)
                        
                        self.xp, self.yp = x1, y1
                    
                    else:
                        self.xp, self.yp = 0, 0

            # Merge canvas with frame
            img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            
            frame = cv2.bitwise_and(frame, img_inv)
            frame = cv2.bitwise_or(frame, self.canvas)

            # Display in Streamlit
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

if __name__ == "__main__":
    app = GesturePainter()
    app.run()
