from djitellopy import Tello
import cv2
import time

print("Connecting to Tello...")
tello = Tello()
tello.connect()
print("Connection successful. Battery:", tello.get_battery())

try:
    print("\n--- TURNING ON VIDEO STREAM ---")
    tello.streamon()
    
    print("Asking for video frame reader... (This is the part that fails)")
    # This line will cause the [Errno 10014] if your firewall is blocking it
    frame_read = tello.get_frame_read() 
    print("Successfully got frame reader.")

    print("\n--- DISPLAYING VIDEO (Press 'q' to quit) ---")
    while True:
        frame = frame_read.frame
        if frame is None:
            continue
            
        cv2.imshow("Tello Video Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"Error: {e}")
    print("This is the expected error if your firewall is on.")

finally:
    print("\nCleaning up...")
    tello.streamoff()
    print("Test complete.")