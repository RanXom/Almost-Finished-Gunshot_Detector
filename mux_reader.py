import RPi.GPIO as GPIO
import time
import numpy as np

# GPIO pin setup for multiplexer control
mux_control_pins = [5, 6, 13]  # Example GPIO control pins
GPIO.setmode(GPIO.BCM)
for pin in mux_control_pins:
    GPIO.setup(pin, GPIO.OUT)

# GPIO pin setup for reading the audio signal (assuming one pin for the output)
audio_signal_pin = 17  # Example GPIO pin for audio signal
GPIO.setup(audio_signal_pin, GPIO.IN)

def read_mux_output(mic_index):
    """Read the signal from the multiplexer for the given microphone index."""
    control_values = [int(x) for x in format(mic_index, '03b')]  # Binary values for control pins
    
    for j, pin in enumerate(mux_control_pins):
        GPIO.output(pin, control_values[j])
    
    time.sleep(0.01)  # Allow time for switching

    # Read digital value from the microphone
    # You may want to sample this multiple times to get a better representation
    signal = [GPIO.input(audio_signal_pin) for _ in range(16000)]  # Sample 16000 times
    return np.array(signal)  # Return as a numpy array
