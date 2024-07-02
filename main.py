from flask import Flask, jsonify
import time # For prototyping
from flask_cors import CORS
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import librosa

from pydub import AudioSegment

import serial.tools.list_ports
import time
import os

# This script located at?
# Required for making absolute path of this script
script_dir = os.path.dirname(__file__)

# Make "data" folder to contain audio recording
if not os.path.exists("data"): 
	os.makedirs("data") 

DATA = os.path.join('data')
NOISE_PROFILE = os.path.join('profile.wav')

def convert_audio(file_path):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Check if the audio is already 16-bit
    if audio.sample_width == 2:
        print(f"Audio {file_path} is already 16-bit.")
        return

    # Set the sample width to 2 bytes (16-bit)
    audio = audio.set_sample_width(2)

    # Save the 16-bit audio to the same file
    audio.export(file_path, format="wav")  # Specify the format as needed
    print(f"Converted {file_path} to 16-bit.")

def get_mean_amplitude_from_audio(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)
    # Calculate the DC offset
    dc_offset = np.mean(audio)

    # Subtract the DC offset from the audio signal
    audio = audio - dc_offset

    # Calculate the mean amplitude
    rms_amplitude = np.sqrt(np.mean(np.square(audio)))

    # Define an arbitrary reference level
    ref_level = 0.1

    if rms_amplitude > 0:  # Ensure non-zero to avoid log(0)
        spl = 10 * np.log10(rms_amplitude / ref_level)
    else:
        spl = -np.inf  # Log of zero amplitude is minus infinity

    return spl

# Define a function for spectral subtraction
def spectral_subtraction(wav, noise_profile, frame_length=1024, frame_step=512):
    # Compute the STFT of both the signal and the noise
    wav_stft = tf.signal.stft(wav, frame_length=frame_length, frame_step=frame_step)
    noise_stft = tf.signal.stft(noise_profile, frame_length=frame_length, frame_step=frame_step)

    # Estimate the mean noise spectrogram
    mean_noise_amp = tf.reduce_mean(tf.abs(noise_stft), axis=0)

    # Subtract the mean noise amplitude from the signal's magnitude
    signal_mag = tf.abs(wav_stft) - mean_noise_amp
    signal_mag = tf.maximum(signal_mag, 0)  # Ensure the spectrum is non-negative

    # Reconstruct the signal using the original phase
    signal_phase = tf.math.angle(wav_stft)
    
    # Reconstruct the complex-valued STFT using magnitude and phase
    signal_real = signal_mag * tf.cos(signal_phase)
    signal_imag = signal_mag * tf.sin(signal_phase)
    denoised_stft = tf.complex(signal_real, signal_imag)
    
    return denoised_stft

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 4000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=4000)
    return wav

def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:7000]

    # Load noise profile
    noise_profile = load_wav_16k_mono(NOISE_PROFILE)
    noise_profile = noise_profile[:7000]
    
    zp_profile = tf.zeros([7000] - tf.shape(noise_profile), dtype=tf.float32)
    noise_profile = tf.concat([zp_profile, noise_profile],0)

    zero_padding = tf.zeros([7000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)

    # Perform spectral subtraction
    denoised_stft = spectral_subtraction(wav, noise_profile, 64, 30)
    
    spectrogram = tf.abs(denoised_stft)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

def convert_folder(folder_path):
    # Loop over all files in the directory
    for filename in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        # Check if file is an audio file (e.g., ends with .wav)
        if file_path.lower().endswith(".wav"):
            convert_audio(file_path)

def calculate_amplitude(folder_path):
    amplitudes = []
    # Loop over all files in the directory
    for filename in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        # Check if file is an audio file (e.g., ends with .wav)
        if file_path.lower().endswith(".wav"):
            amplitudes.append(get_mean_amplitude_from_audio(file_path))
    
    return amplitudes

loaded_model = tf.keras.models.load_model('leak.keras')

######################################################
# Arduino Preparation and opening serial port
arduino = serial.Serial()

portVar = "/dev/ttyUSB0"

arduino.baudrate = 115200
arduino.port = portVar
arduino.open()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/data', methods=['GET'])
def get_data():
    arduino.write(bytes("RECORD", 'utf-8')) 
    time.sleep(0.05)
    
    num = 0
		
    while True:
        if arduino.in_waiting:
			# Readline for splitting serial communication with \n (0a) per packet
            packet = arduino.readline() 
            msg = packet.decode('utf8', "ignore") # "ignore" means non-UTF byte goes through without decoding

			# If there is "ERROR" or "EVENT" in serial message, print it
            if (msg.find("ERROR", 0, 5) != -1 or msg.find("EVENT", 0, 5) != -1):
                print(msg, end="")

				# If there is "wav" in serial message, increase num by 1, and get NOW time unix epoch
				# Also opening file for writing
                if (msg.find("wav") != -1):
                    if (num != 0):
                        file.close()
                    num = num + 1
                    timeStr = str(int(time.time()))
                    abs_file_path = os.path.join(script_dir, "data/Mic " + str(num) + ".wav")
                    file = open(abs_file_path, "wb") 

				# If there is "Done Sending Data to PC", break the while loop
                if (msg.find("Done Sending Data to PC") != -1):
                    break
			# Else, write the file
            else:
                file.write(packet)

	# Truncate 0a (\n , newline) byte from the file
    for i in range(1,4):
        abs_file_path = os.path.join(script_dir, "data/Mic " + str(num) + ".wav")
        with open(abs_file_path, "r+b") as f:
			# Move the file pointer to the end of the file.
            f.seek(0, 2)
			# Truncate the file to the previous byte.
            f.truncate(f.tell() - 1)
    
    convert_folder(DATA)

    data = tf.data.Dataset.list_files(DATA + '/*.wav')
    data = data.map(lambda x: preprocess(x, 0))
    data = data.batch(3)

    predictions = loaded_model.predict(data)
    predictions = [item for sublist in predictions for item in sublist]
    predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]
    prediction = predictions.count(1) >= 1
    print(predictions)

    amplitudes = calculate_amplitude(DATA)

    L = 0.2
    AB = amplitudes[0] - amplitudes[1]
    BC = amplitudes[1] - amplitudes[2]

    theta = 0

    OB = ( np.sqrt(2) * L ) / np.sqrt( 10**(-0.1*AB) + 10**(0.1*BC) - 2 )
    theta = np.arccos( (np.sqrt(2) * (10**(0.1*BC) - 10**(-0.1*AB))) / (4 * np.sqrt( 10**(-0.1*AB) + 10**(0.1*BC) - 2 ) ) )

    y = (OB * np.sin(theta)) * 100
    x = (L - (OB * np.cos(theta))) * 100

    print(amplitudes)
    print(AB, BC)
    print(x, y)

    response_data = {
        "leak": prediction,
        "x": x,
        "y": y
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
