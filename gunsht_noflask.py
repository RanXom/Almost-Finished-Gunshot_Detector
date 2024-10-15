import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, lfilter
import pickle
import librosa
import queue
from display_radar import display_radar
from localisation import locate_gunshot, MIC_POSITIONS  # Import localization functions

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define your class labels
classes = ['gun_shot', 'other_sound']

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback function to process audio input."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter to the audio data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def normalize_audio(data):
    """Normalize the audio data to range [-1, 1]."""
    return data / np.max(np.abs(data))

def predict(audio_data):
    """Predict class from audio data."""
    temp_file_path = 'temp.wav'
    sf.write(temp_file_path, audio_data, 16000)

    audiodata, sample_rate = librosa.load(temp_file_path, res_type='kaiser_fast')
    mels = np.mean(librosa.feature.melspectrogram(y=audiodata, sr=sample_rate).T, axis=0)
    X = np.array(mels).reshape(1, -1)

    classid = np.argmax(model.predict(X))
    predicted_class = classes[classid] if classid < len(classes) else "Unknown"

    return predicted_class

def process_audio():
    """Continuously process audio input and detect gunshots."""
    fs = 16000  # Sampling frequency
    lowcut = 500.0  # Low cut frequency for bandpass
    highcut = 3000.0  # High cut frequency for bandpass
    threshold = 0.05  # Set a threshold level

    print("Listening for gunshots... (Press Ctrl+C to stop)")
    toa_list = []  # To store time of arrivals
    while True:
        try:
            if not audio_queue.empty():
                audiodata = audio_queue.get()
                filtered_data = bandpass_filter(audiodata.flatten(), lowcut, highcut, fs)
                normalized_data = normalize_audio(filtered_data)

                # Apply thresholding
                if np.max(np.abs(normalized_data)) > threshold:
                    predicted_class = predict(normalized_data)

                    if predicted_class == "gun_shot":
                        print("Gunshot detected!")

                        # Assuming you can measure ToA for 6 microphones; here we simulate ToA for demo
                        toa_list = [0.0012, 0.0014, 0.0013, 0.0011, 0.0015, 0.0016]  # Replace with actual ToA values

                        # Check if we have enough microphones
                        if len(toa_list) >= 3:  # Need at least 3 ToA values for localization
                            location = locate_gunshot(toa_list)
                            print("Gunshot location:", location)

                            # Display radar (you may want to implement this in a separate function)
                            display_radar(location)
                        else:
                            print("Insufficient microphones for localization.")
                        break  # Stop listening after detecting the gunshot
        except KeyboardInterrupt:
            print("Stopped listening.")
            break

if __name__ == "__main__":
    # Start the audio stream
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        process_audio()
