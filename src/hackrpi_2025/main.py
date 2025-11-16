import rtmidi
import time
import cv2
import mediapipe as mp
import sys
import os
import warnings
import math
import numpy as np               
from djitellopy import Tello   


def list_video_devices():
    print("Available video devices:")
    available_devices = []
    # Suppress OpenCV warnings temporarily
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for i in range(10):  # Check first 10 devices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        # Try to get device name if available
                        backend = cap.getBackendName()
                        print(f"  Device {i}: Available (backend: {backend})")
                        available_devices.append(i)
                    cap.release()
            except Exception:
                pass
    return available_devices


def get_hand_size(hand_landmarks, mp_hands):
    landmarks = hand_landmarks.landmark
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    return ((wrist.x - middle_mcp.x) ** 2 + (wrist.y - middle_mcp.y) ** 2) ** 0.5


def get_hand_closedness(hand_landmarks, mp_hands):
    # Get landmarks
    landmarks = hand_landmarks.landmark

    hand_size = get_hand_size(hand_landmarks, mp_hands)
    max_distance = hand_size * 0.75

    # Define finger tip, MCP (metacarpophalangeal), and PIP (proximal interphalangeal) joints
    finger_data = [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_MCP),
        (
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
        ),
        (
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        ),
        (
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
        ),
        (
            mp_hands.HandLandmark.PINKY_TIP,
            mp_hands.HandLandmark.PINKY_MCP,
            mp_hands.HandLandmark.PINKY_PIP,
        ),
    ]

    finger_closedness_values = []

    for tip_idx, mcp_idx, pip_idx in finger_data:
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]

        distance = ((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2) ** 0.5
        tip_below_pip = tip.y > pip.y
        tip_below_mcp = tip.y > mcp.y

        normalized_distance = min(distance / max_distance, 1.0)
        closedness = 1.0 - normalized_distance

        if tip_below_pip:
            closedness = max(closedness, 0.7)
        
        if tip_below_mcp:
            closedness = max(closedness, 0.9)

        closedness = max(0.0, min(1.0, closedness))
        finger_closedness_values.append(closedness)

    hand_closedness = sum(finger_closedness_values) / len(finger_closedness_values)

    return hand_closedness


def get_hand_rotation(hand_landmarks, mp_hands):
    landmarks = hand_landmarks.landmark
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) - 85
    if angle_deg < 0:
        angle_deg += 360
    elif angle_deg > 360:
        angle_deg %= 360
    rotation = angle_deg / 360.0
    return rotation


def select_video_device():
    if len(sys.argv) > 1:
        try:
            device_id = int(sys.argv[1])
            print(f"Using video device {device_id} (from command line argument)")
            return device_id
        except ValueError:
            print(f"Invalid device ID: {sys.argv[1]}. Must be a number.")
            sys.exit(1)

    env_device = os.getenv("VIDEO_DEVICE")
    if env_device:
        try:
            device_id = int(env_device)
            print(
                f"Using video device {device_id} (from VIDEO_DEVICE environment variable)"
            )
            return device_id
        except ValueError:
            print(f"Invalid device ID in VIDEO_DEVICE: {env_device}. Must be a number.")
            sys.exit(1)

    print("No device specified. Listing available devices:")
    list_video_devices()
    return int(input("Enter device ID: "))

def _map_value(x, in_min, in_max, out_min, out_max):
    """Helper function to map a value from one range to another."""
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

mode = "hand" # Default
if len(sys.argv) > 1 and sys.argv[1].lower() == "drone":
    mode = "drone"
    sys.argv.pop(1) 
print(f"Running in {mode} mode.")

tello = None
# --- MODIFIED: Go back to using frame_read ---
frame_read = None 
frame_width = 960 # default
frame_height = 720 # default

# OpenCV Color Tuning
LOWER_RED_1 = np.array([0, 150, 100])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([160, 150, 100])
UPPER_RED_2 = np.array([180, 255, 255])

if mode == "drone":
    try:
        tello = Tello()
        tello.connect()
        print("Tello Connected. Battery:", tello.get_battery())
        tello.streamon()
        
        # --- MODIFIED: Use get_frame_read() ---
        frame_read = tello.get_frame_read()
        
        if frame_read is None:
            raise Exception("Failed to get Tello frame reader.")
            
        time.sleep(2) # Give stream time to start
        
        frame = frame_read.frame
        if frame is not None:
            frame_height, frame_width = frame.shape[:2]
        else:
             print("Warning: Could not get frame dimensions, using defaults.")
        # --- END MODIFICATIONS ---
        
        print(f"Video stream initiated: {frame_width}x{frame_height}")
        tello.takeoff()
        tello.hover()
    except Exception as e:
        print(f"Failed to initialize Tello: {e}")
        if tello:
            tello.land()
        sys.exit(1)
else:
    # Hand tracking setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    device_id = select_video_device()
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Error: Could not open video device {device_id}")
        list_video_devices()
        sys.exit(1)

midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()
if not available_ports:
    print("Error: No MIDI ports found!")
    sys.exit(1)
midiout.open_port(0)
print("Opened port", available_ports[0])

running = True
knob_values = [0, 0, 0, 0]
dt = 1 / 30
x = 0
y = 0
hand_openness = 0.0
hand_rotation = 0.0

padding = 0.12
vertical_padding = 0.2
min_closedness = 0.15
max_closedness = 0.7
min_rotation = 0.42
max_rotation = 0.58

while running:
    try:
        if mode == "hand":
            # --- HAND TRACKING LOGIC ---
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]
            results = hands.process(rgb_frame)

            hand_closedness = 0.0
            hand_rotation = 0.0

            if results.multi_hand_landmarks:
                sizes = [
                    get_hand_size(hand_landmarks, mp_hands)
                    for hand_landmarks in results.multi_hand_landmarks
                ]
                max_index = sizes.index(max(sizes))
                hand_landmarks = results.multi_hand_landmarks[max_index]

                hand_closedness = get_hand_closedness(hand_landmarks, mp_hands)
                hand_closedness = max(min_closedness, min(max_closedness, hand_closedness))
                hand_closedness = (hand_closedness - min_closedness) / (
                    max_closedness - min_closedness
                )
                hand_openness = 1.0 - hand_closedness

                hand_rotation = get_hand_rotation(hand_landmarks, mp_hands)
                hand_rotation = max(min_rotation, min(max_rotation, hand_rotation))
                hand_rotation = (hand_rotation - min_rotation) / (
                    max_rotation - min_rotation
                )
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_tip
                ]

                x = max(padding, min(1.0 - padding, index_finger_tip.x))
                y = max(vertical_padding, min(1.0 - vertical_padding, index_finger_tip.y))
                x -= padding
                y -= vertical_padding
                x /= 1.0 - 2 * padding
                y /= 1.0 - 2 * vertical_padding

                tip_x = int(index_finger_tip.x * frame_width)
                tip_y = int(index_finger_tip.y * frame_height)
                circle_color = (255, 0, 0)
                cv2.circle(frame, (tip_x, tip_y), 32, circle_color, 8)

                padding_x1 = int(frame_width * padding)
                padding_y1 = int(frame_height * vertical_padding)
                padding_x2 = int(frame_width * (1 - padding))
                padding_y2 = int(frame_height * (1 - vertical_padding))
                cv2.rectangle(
                    frame,
                    (padding_x1, padding_y1),
                    (padding_x2, padding_y2),
                    (0, 255, 0),
                    2,
                )

                hand_state_text = f"X: {x:.2f} Y: {y:.2f} Openness: {hand_openness:.2f} Rotation: {hand_rotation:.2f}"
                cv2.putText(
                    frame,
                    hand_state_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    circle_color,
                    2,
                )
            else:
                hand_openness = 0.0
                hand_rotation = 0.0
                x = 0.0
                y = 0.0

            knob_values[0] = max(0, min(127, int(x * 127)))
            knob_values[1] = max(0, min(127, int(y * 127)))
            knob_values[2] = max(0, min(127, int(hand_openness * 127)))
            knob_values[3] = max(0, min(127, int(hand_rotation * 127)))

            cv2.imshow("Hand Tracking", frame)
            
        else:
            # --- MODIFIED: DRONE TRACKING LOGIC ---
            
            # 1. GET DRONE DATA
            # --- MODIFIED: Read from frame_read ---
            frame = frame_read.frame
            if frame is None:
                continue
            # --- END MODIFICATION ---

            # We use blob area for Z-axis (volume), not height
            # z_cm = tello.get_height() 
            
            # 2. OPENCV POSITIONING
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
            mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
            mask = mask1 + mask2
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            x_norm, y_norm, z_norm = 0.5, 0.5, 0.0 # Default
            debug_area = 0 # for display
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                
                # --- KEY CHANGE: Use blob area for Z-axis (Volume) ---
                area = cv2.contourArea(c)
                debug_area = area # for display
            
                if M["m00"] > 0:
                    x_px = int(M["m10"] / M["m00"])
                    y_px = int(M["m01"] / M["m00"])
                    x_norm = x_px / frame_width
                    y_norm = y_px / frame_height
                    cv2.circle(frame, (x_px, y_px), 10, (0, 255, 0), 2)
                
                # --- Map blob area to 0.0 - 1.0 volume ---
                # !!! YOU MUST TUNE these min/max area values!
                z_norm = _map_value(area, 5000, 80000, 0.0, 1.0) 
                z_norm = max(0.0, min(1.0, z_norm)) # Clamp
                                
            # 3. POPULATE KNOB VALUES
            knob_values[0] = max(0, min(127, int(x_norm * 127))) # X-Axis
            knob_values[1] = max(0, min(127, int(y_norm * 127))) # Y-Axis
            knob_values[2] = max(0, min(127, int(z_norm * 127))) # Z-Axis (Volume)
            knob_values[3] = 0 # Knob 3 is unused
            
            # 4. DEBUG VIEW
            debug_text = f"X: {x_norm:.2f} Y: {y_norm:.2f} Z (Area): {debug_area}"
            cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Tello Drone Positioning", frame)
            
        # --- COMMON LOGIC ---
        
        midiout.send_message([0xB0, 0, knob_values[0]])
        midiout.send_message([0xB0, 1, knob_values[1]])
        midiout.send_message([0xB0, 2, knob_values[2]])
        midiout.send_message([0xB0, 3, knob_values[3]])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

        time.sleep(dt)
        
    except KeyboardInterrupt:
        running = False
    except Exception as e:
        print("Error:", e)
        time.sleep(1) 

# --- MODIFIED: Cleanup logic ---
if mode == "hand":
    if 'cap' in locals() and cap.isOpened():
        cap.release()
else:
    if tello:
        print("Landing drone...")
        tello.land()
        tello.streamoff()
    # No video_capture object to release

cv2.destroyAllWindows()
midiout.close_port()
print("Script finished.")