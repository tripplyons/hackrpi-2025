import cv2
import numpy as np
import time
from djitellopy import Tello

class DroneController:
    """
    Handles all Tello connection, takeoff, video processing,
    and positioning logic.
    """
    
    def __init__(self):
        print("Initializing DroneController...")
        # --- OpenCV Color Tuning ---
        # Tune these values for your red marker!
        self.LOWER_RED_1 = np.array([0, 150, 100])
        self.UPPER_RED_1 = np.array([10, 255, 255])
        self.LOWER_RED_2 = np.array([160, 150, 100])
        self.UPPER_RED_2 = np.array([180, 255, 255])
        # ---------------------------

        self.tello = Tello()
        self.tello.connect()
        print("Tello Connected. Battery:", self.tello.get_battery())

        self.tello.streamon()
        self.tello.set_video_direction(Tello.CAMERA_DOWNWARD)
        
        self.frame_read = self.tello.get_frame_read()
        time.sleep(2) # Give stream time to start

        frame = self.frame_read.frame
        self.frame_height, self.frame_width = frame.shape[:2]
        print(f"Video stream initiated: {self.frame_width}x{self.frame_height}")

        self.tello.takeoff()
        self.tello.hover()
        
    def _map_value(self, x, in_min, in_max, out_min, out_max):
        """Helper function to map a value from one range to another."""
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def get_normalized_position(self):
        """
        Gets one frame, processes it, and returns the (x, y, z) position.
        Returns:
            (x_norm, y_norm, z_norm), should_stop (bool)
        """
        
        # 1. GET Z-AXIS (Height)
        z_cm = self.tello.get_height()
        
        # 2. GET VIDEO FRAME (for X/Y axis)
        frame = self.frame_read.frame
        if frame is None:
            return (0.5, 0.5, 0.0), False # Return default if frame is bad

        # 3. OPENCV POSITIONING
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.LOWER_RED_1, self.UPPER_RED_1)
        mask2 = cv2.inRange(hsv, self.LOWER_RED_2, self.UPPER_RED_2)
        mask = mask1 + mask2
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Default values
        x_norm, y_norm = 0.5, 0.5 # Default to center
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                x_px = int(M["m10"] / M["m00"])
                y_px = int(M["m01"] / M["m00"])
                
                # Normalize to 0.0 - 1.0
                x_norm = x_px / self.frame_width
                y_norm = y_px / self.frame_height
                
                # Draw debug circle
                cv2.circle(frame, (x_px, y_px), 10, (0, 255, 0), 2)
        
        # Normalize Z (height)
        z_norm = self._map_value(z_cm, 30, 150, 0.0, 1.0)
        z_norm = max(0.0, min(1.0, z_norm)) # Clamp
        
        # 4. SHOW DEBUG VIEW
        debug_text = f"X: {x_norm:.2f} Y: {y_norm:.2f} Z: {z_norm:.2f}"
        cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Tello Drone Positioning", frame)
        
        # 5. CHECK FOR 'q' KEY TO QUIT
        should_stop = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            should_stop = True

        return (x_norm, y_norm, z_norm), should_stop

    def land_and_cleanup(self):
        """Lands the drone, stops the stream, and closes windows."""
        print("Landing and cleaning up...")
        self.tello.land()
        self.tello.streamoff()
        cv2.destroyAllWindows()