from djitellopy import Tello
import time

print("Connecting to Tello...")
tello = Tello()
tello.connect()

print("Connection successful. Battery:", tello.get_battery())

try:
    print("\n--- TAKING OFF ---")
    tello.takeoff()
    print("Takeoff successful. Hovering for 2 seconds...")
    
    # Wait for 2 seconds
    time.sleep(2)
    
    print("\n--- LANDING ---")
    tello.land()
    print("Landing successful.")

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"Error: {e}")
    print("Attempting to land...")
    tello.land() # Safety land
finally:
    print("\nTest complete.")