import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
import sounddevice as sd
from localisation import locate_gunshot, create_plotly_3d_plot, MIC_POSITIONS 
from flask import Flask, request, jsonify, render_template
from display_radar import display_radar

# Load your pre-trained gunshot detection model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load UrbanSound8K dataset for class labels
audiofiles = pd.read_csv(r'C:\Users\beide\OneDrive\Documents\National Hackathon\Gunshot-detection\UrbanSound8K.csv')
classes = dict(zip(audiofiles['classID'], audiofiles['class']))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index2.html')

def detect_gunshot(indata, sample_rate=22050):
    """
    Process microphone input and detect gunshots.
    """
    audiodata = np.array(indata).flatten()
    mels = np.mean(librosa.feature.melspectrogram(y=audiodata, sr=sample_rate).T, axis=0)
    X = np.array(mels).reshape(1, -1)

    classid = np.argmax(model.predict(X))
    predicted_class = classes[classid] if classid < len(classes) else "Unknown"
    
    return predicted_class

def audio_callback(indata, frames, time, status):
    """
    Callback function for real-time microphone input.
    """
    predicted_class = detect_gunshot(indata)
    
    if predicted_class == "gun_shot":
        toa = get_time_of_arrival_from_mics()  # Function to compute ToA from mic array
        location = locate_gunshot(toa)
        display_radar(location)  # Display location on radar

@app.route('/predict', methods=['POST'])
def predict():
    # Code to upload and classify audio files (existing code)
    pass

def get_time_of_arrival_from_mics():
    # Implement your logic to calculate ToA for each microphone here
    toa = [0.0012, 0.0014, 0.0013, 0.0011, 0.0015, 0.0016]  # Example ToA
    return toa

if __name__ == '__main__':
    # Start microphone stream and Flask server
    sample_rate = 22050
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
        app.run(debug=True)
