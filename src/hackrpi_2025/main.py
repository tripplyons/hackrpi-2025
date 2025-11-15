import rtmidi
import time
import cv2
import mediapipe as mp
import sys
import os
import warnings
import math


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


def get_hand_closedness(hand_landmarks, mp_hands):
    # Get landmarks
    landmarks = hand_landmarks.landmark

    # Calculate hand size based on wrist to middle finger MCP distance (2D only)
    # This distance is relatively stable regardless of finger position or rotation
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    hand_size = ((wrist.x - middle_mcp.x) ** 2 + (wrist.y - middle_mcp.y) ** 2) ** 0.5

    max_distance = hand_size * 0.75

    # Define finger tip and MCP (metacarpophalangeal) joint pairs
    finger_pairs = [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
        ),
        (
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        ),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP),
    ]

    finger_closedness_values = []

    for tip_idx, mcp_idx in finger_pairs:
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]

        # Calculate Euclidean distance in 2D space (x, y only) to be rotation-invariant
        # The z-coordinate (depth) varies with rotation and doesn't indicate finger bend
        distance = ((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2) ** 0.5

        # Normalize distance to 0-1 range and invert (smaller distance = more closed = higher value)
        # Use adaptive max_distance based on hand size
        normalized_distance = min(distance / max_distance, 1.0)
        closedness = 1.0 - normalized_distance

        # Clamp to 0-1 range
        closedness = max(0.0, min(1.0, closedness))
        finger_closedness_values.append(closedness)

    # Average across all fingers to get overall hand closedness
    hand_closedness = sum(finger_closedness_values) / len(finger_closedness_values)

    return hand_closedness


def get_hand_rotation(hand_landmarks, mp_hands):
    landmarks = hand_landmarks.landmark

    # Get wrist and middle finger MCP positions
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Calculate vector from wrist to middle finger MCP
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y

    # Calculate angle in radians (atan2 gives angle from -π to π)
    angle_rad = math.atan2(dy, dx)

    # Convert to degrees (0 to 360)
    angle_deg = math.degrees(angle_rad) - 85
    if angle_deg < 0:
        angle_deg += 360
    elif angle_deg > 360:
        angle_deg %= 360

    # Normalize to 0-1 range (0 degrees = 0.0, 360 degrees = 1.0)
    rotation = angle_deg / 360.0

    return rotation


def select_video_device():
    # Check command line argument
    if len(sys.argv) > 1:
        try:
            device_id = int(sys.argv[1])
            print(f"Using video device {device_id} (from command line argument)")
            return device_id
        except ValueError:
            print(f"Invalid device ID: {sys.argv[1]}. Must be a number.")
            sys.exit(1)

    # Check environment variable
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

    # Default to device 0, but list available devices
    print("No device specified. Listing available devices:")
    list_video_devices()
    return int(input("Enter device ID: "))


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV camera with manual device selection
device_id = select_video_device()
cap = cv2.VideoCapture(device_id)

if not cap.isOpened():
    print(f"Error: Could not open video device {device_id}")
    print("\nAvailable devices:")
    list_video_devices()
    sys.exit(1)

midiout = rtmidi.MidiOut()

available_ports = midiout.get_ports()

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
min_closedness = 0.2
max_closedness = 0.7
min_rotation = 0.42
max_rotation = 0.58

while running:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame horizontally (flip left-right)
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get frame dimensions for reference
        frame_height, frame_width = frame.shape[:2]

        # Process frame for hand landmarks
        results = hands.process(rgb_frame)

        hand_closedness = 0.0
        hand_rotation = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate how closed the hand is (0.0 = open, 1.0 = closed)
                hand_closedness = get_hand_closedness(hand_landmarks, mp_hands)
                hand_closedness = max(
                    min_closedness, min(max_closedness, hand_closedness)
                )
                hand_closedness = (hand_closedness - min_closedness) / (
                    max_closedness - min_closedness
                )
                hand_openness = 1.0 - hand_closedness

                # Calculate hand rotation (0.0 to 1.0)
                hand_rotation = get_hand_rotation(hand_landmarks, mp_hands)
                hand_rotation = max(min_rotation, min(max_rotation, hand_rotation))
                hand_rotation = (hand_rotation - min_rotation) / (
                    max_rotation - min_rotation
                )
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                index_finger_tip = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]

                x = max(padding, min(1.0 - padding, index_finger_tip.x))
                y = max(
                    vertical_padding, min(1.0 - vertical_padding, index_finger_tip.y)
                )
                x -= padding
                y -= vertical_padding
                x /= 1.0 - 2 * padding
                y /= 1.0 - 2 * vertical_padding

                # Convert normalized coordinates to pixel coordinates
                tip_x = int(index_finger_tip.x * frame_width)
                tip_y = int(index_finger_tip.y * frame_height)

                circle_color = (255, 0, 0)
                cv2.circle(frame, (tip_x, tip_y), 32, circle_color, 8)

                # Draw padding boundary rectangle (convert to integers)
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

                # Display hand closedness and rotation on frame
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

        midiout.send_message([0xB0, 0, knob_values[0]])
        midiout.send_message([0xB0, 1, knob_values[1]])
        midiout.send_message([0xB0, 2, knob_values[2]])
        midiout.send_message([0xB0, 3, knob_values[3]])

        # Display the frame
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

        time.sleep(dt)
    except KeyboardInterrupt:
        running = False
    except Exception as e:
        print("Error:", e)
        running = False

cap.release()
cv2.destroyAllWindows()
midiout.close_port()
