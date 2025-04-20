import cv2
import mediapipe as mp
import time
from twilio.rest import Client
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
 
ACCOUNT_SID = ''
AUTH_TOKEN = ''
FROM_PHONE = ''
TO_PHONE = ''

client = Client(ACCOUNT_SID, AUTH_TOKEN)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hand_closed_count = 0
reset_interval = 1
last_reset_time = time.time()
hand_was_closed = False
cap = cv2.VideoCapture(0)  # Camera starts right away


def is_thumb_hidden(hand_landmarks):
    """Check if the thumb is hidden (a signal for SOS)."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    is_hidden = thumb_tip.x < index_mcp.x and thumb_tip.y > index_mcp.y
    return is_hidden


def send_sos_message():
    """Function to send SOS alert message via Twilio."""
    try:
        message = client.messages.create(
            body="SOS detected!",
            from_=FROM_PHONE,
            to=TO_PHONE
        )
        print(f"Message sent: {message.sid}")
    except Exception as e:
        print(f"Failed to send message: {e}")


def start_detection():
    """Start the hand detection process."""
    global hand_closed_count, last_reset_time, hand_was_closed

    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not accessible!")
        return

    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            current_time = time.time()

            if current_time - last_reset_time > reset_interval:
                hand_closed_count = 0
                last_reset_time = current_time

            hand_currently_closed = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb_hidden = is_thumb_hidden(hand_landmarks)
                    if thumb_hidden:
                        hand_closed_count += 1
                        last_reset_time = current_time
                        hand_currently_closed = True
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_was_closed = hand_currently_closed

            cv2.putText(image, f'Count: {hand_closed_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if hand_closed_count >= 2:
                cv2.putText(image, 'SOS', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                send_sos_message()  # Sends SOS alert

            cv2.imshow('Hand State Detection', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def launch_gui():
    """Launches a colorful GUI with Start Detection functionality."""
    root = tk.Tk()
    root.title("SOS Hand Detection")
    root.geometry("600x400")

    # Styling the GUI with colors
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background="#3E4149")
    style.configure("TLabel", background="#3E4149", foreground="white")
    style.configure("TButton", background="#6C757D", foreground="white")
    
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)

    # Start button
    start_button = ttk.Button(frame, text="Start Detection", command=start_detection)
    start_button.grid(row=0, column=0, padx=20, pady=20)

    # Message to indicate exit
    exit_label = ttk.Label(frame, text="Press 'q' in the camera window to quit.")
    exit_label.grid(row=1, column=0, padx=20, pady=10)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
