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

# --- MODIFIED: Kept only the Up/Down gain ---
P_GAIN_UD = 0.7   # Gain for up/down (altitude)

# --- NEW: Altitude Range (3ft to 4ft in cm) ---
MIN_ALTITUDE_CM = 91  # 3 feet
MAX_ALTITUDE_CM = 122 # 4 feet
# ---

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


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle at p2 formed by points p1-p2-p3.
    Returns angle in radians (0 to π).
    """
    # Create vectors from p2 to p1 and p2 to p3
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    # Calculate dot product and magnitudes
    dot_product = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return math.pi

    # Calculate angle using arccos
    cos_angle = dot_product / (mag1 * mag2)
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.acos(cos_angle)

    return angle


def get_hand_closedness(hand_landmarks, mp_hands):
    # Get landmarks
    landmarks = hand_landmarks.landmark

    # Define finger joints
    finger_data = [
        (
            mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.THUMB_IP,
            mp_hands.HandLandmark.THUMB_MCP,
            mp_hands.HandLandmark.THUMB_CMC,
        ),
        (
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.INDEX_FINGER_DIP,
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
        ),
        (
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        ),
        (
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_DIP,
            mp_hands.HandLandmark.RING_FINGER_MCP,
        ),
        (
            mp_hands.HandLandmark.PINKY_TIP,
            mp_hands.HandLandmark.PINKY_PIP,
            mp_hands.HandLandmark.PINKY_DIP,
            mp_hands.HandLandmark.PINKY_MCP,
        ),
    ]

    finger_closedness_values = []

    for tip_idx, pip_idx, dip_idx, mcp_idx in finger_data:
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        dip = landmarks[dip_idx]
        mcp = landmarks[mcp_idx]

        if tip_idx == mp_hands.HandLandmark.THUMB_TIP:
            angle_ip = calculate_angle(dip, pip, tip)  # MCP-IP-TIP
            normalized_angle = 1.0 - (angle_ip - math.pi / 2) / (math.pi / 2)
            closedness = max(0.0, min(1.0, normalized_angle))
        else:
            angle_pip = calculate_angle(mcp, pip, dip)  # Angle at PIP joint
            angle_dip = calculate_angle(pip, dip, tip)  # Angle at DIP joint
            avg_angle = (angle_pip + angle_dip) / 2.0
            normalized_angle = 1.0 - (avg_angle - math.pi / 2) / (math.pi / 2)
            closedness = max(0.0, min(1.0, normalized_angle))

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

def clamp(n, minn, maxn):
    """Helper function to clamp a value."""
    return max(min(maxn, n), minn)


mode = "hand"  # Default
if len(sys.argv) > 1 and sys.argv[1].lower() == "drone":
    mode = "drone"
    # Remove "drone" arg so it doesn't confuse select_video_device
    sys.argv.pop(1)
print(f"Running in {mode} mode.")

tello = None
video_capture = None # Use this instead of frame_read
frame_width = 960  # default
frame_height = 720  # default

# --- MODIFIED: Initialize MediaPipe for BOTH modes ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# ---

if mode == "drone":
    try:
        tello = Tello()
        tello.connect()
        print("Tello Connected. Battery:", tello.get_battery())
        tello.streamon()
        
        # --- THIS IS THE FIREWALL FIX ---
        # Get the stream URL and open it with OpenCV
        stream_url = tello.get_stream_url()
        print(f"Opening video stream at: {stream_url}")
        video_capture = cv2.VideoCapture(stream_url)
        # --- END FIX ---
        
        if not video_capture.isOpened():
            raise Exception("Failed to get Tello video capture with OpenCV.")
        
        # Get frame dimensions from the video capture
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_height == 0 or frame_width == 0:
            print("Warning: Could not get frame dimensions, using defaults.")
            frame_width = 960
            frame_height = 720
        
        print(f"Video stream initiated: {frame_width}x{frame_height}")
        tello.takeoff()
        tello.hover()
    except Exception as e:
        print(f"Failed to initialize Tello: {e}")
        if tello:
            tello.land()
        sys.exit(1)
else:
    # Hand tracking setup (webcam)
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
    if mode == "drone" and tello:
        tello.land()
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
min_closedness = 0.05
max_closedness = 0.75
min_rotation = 0.42
max_rotation = 0.58


while running:
    try:
        # --- MODIFIED: Get frame from correct source ---
        if mode == "hand":
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1) # Flip webcam
        else: # "drone" mode
            ret, frame = video_capture.read()
            if not ret or frame is None:
                print("FRAME IS NONE")
                time.sleep(dt)
                continue
            # Do NOT flip the drone's camera feed
        
        # --- COMMON LOGIC ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # Optimize
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        frame_height, frame_width = frame.shape[:2]

        # Calculate bounding box in pixels
        padding_x1 = int(frame_width * padding)
        padding_y1 = int(frame_height * vertical_padding)
        padding_x2 = int(frame_width * (1 - padding))
        padding_y2 = int(frame_height * (1 - vertical_padding))
        
        # Draw bounding box
        cv2.rectangle(
            frame,
            (padding_x1, padding_y1),
            (padding_x2, padding_y2),
            (0, 255, 0), # Green box
            2,
        )

        hand_closedness = 0.0
        hand_rotation = 0.0
        
        debug_text = "No Hand Found"
        
        # --- NEW: Initialize drone speeds to 0 ---
        ud_speed = 0

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
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]
            
            # Get normalized 0.0-1.0 coords
            x_norm = index_finger_tip.x
            y_norm = index_finger_tip.y
            
            # Get pixel coords
            tip_x = int(x_norm * frame_width)
            tip_y = int(y_norm * frame_height)
            
            # Calculate x, y relative to the *inside* of the box
            x = (tip_x - padding_x1) / (padding_x2 - padding_x1)
            x = clamp(x, 0.0, 1.0) # Clamp to 0-1
            y = (tip_y - padding_y1) / (padding_y2 - padding_y1)
            y = clamp(y, 0.0, 1.0) # Clamp to 0-1
            
            # --- MODIFIED: DRONE "CONDUCTOR" LOGIC ---
            if mode == "drone":
                # Map hand openness (0-1) to target altitude in cm
                target_altitude = _map_value(hand_openness, 0.0, 1.0, MIN_ALTITUDE_CM, MAX_ALTITUDE_CM)
                # Get current altitude
                current_altitude = tello.get_height() # This uses the TOF sensor
                
                # Calculate altitude error and speed
                altitude_error = target_altitude - current_altitude
                ud_speed = int(altitude_error * P_GAIN_UD)
                ud_speed = clamp(ud_speed, -40, 40) # Clamp speed
                
                knob_values[2] = max(0, min(127, int(hand_openness * 127)))                
                tello.send_rc_control(0, 0, ud_speed, 0) # (lr, fb, ud, yaw)
                
                debug_text = f"TargetAlt: {target_altitude:.0f}cm | CurrentAlt: {current_altitude}cm"
                
            else:
                circle_color = (255, 0, 0)
                cv2.circle(frame, (tip_x, tip_y), 32, circle_color, 8)
                debug_text = f"X: {x:.2f} Y: {y:.2f} Openness: {hand_openness:.2f} Rotation: {hand_rotation:.2f}"
                knob_values[2] = max(0, min(127, int(hand_openness * 127))) # Hand mode uses openness for Knob 2

        else:
            hand_openness = 0.0
            hand_rotation = 0.0
            x = 0.0
            y = 0.0
            
            if mode == "drone":
                tello.send_rc_control(0, 0, 0, 0)
                debug_text = "No Hand Found. Hovering."
                knob_values[2] = 0 # No hand, volume = 0

        
        knob_values[0] = max(0, min(127, int(x * 127)))
        knob_values[1] = max(0, min(127, int(y * 127)))
        # Knob 2 is set differently by each mode now
        knob_values[3] = max(0, min(127, int(hand_rotation * 127)))
        
        # Send MIDI
        midiout.send_message([0xB0, 0, knob_values[0]])
        midiout.send_message([0xB0, 1, knob_values[1]])
        midiout.send_message([0xB0, 2, knob_values[2]])
        midiout.send_message([0xB0, 3, knob_values[3]])
        
        # Show debug text and video
        cv2.putText(
            frame,
            debug_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Hand Tracking", frame)

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
else: # "drone"
    if tello:
        print("Landing drone...")
        # Send one last stop command before landing
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.1)
        tello.land()
        tello.streamoff()
    if video_capture:
        video_capture.release()

if 'midiout' in locals() and midiout:
    midiout.close_port()
    
cv2.destroyAllWindows()
print("Script finished.")