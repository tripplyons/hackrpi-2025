import rtmidi
import time

midiout = rtmidi.MidiOut()

available_ports = midiout.get_ports()

midiout.open_port(0)
print("Opened port", available_ports[0])

running = True
knob_values = [0, 0]
dt = 0.01
x = 0
y = 0

while running:
    try:
        knob_values[0] = int(x * 127)
        knob_values[1] = int(y * 127)
        x += dt
        y += dt
        x %= 1
        y %= 1

        midiout.send_message([0xB0, 0, knob_values[0]])
        midiout.send_message([0xB0, 1, knob_values[1]])

        time.sleep(dt)
    except KeyboardInterrupt:
        running = False
    except Exception as e:
        print("Error:", e)
        running = False

midiout.close_port()
