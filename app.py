from flask import Flask, request, jsonify, send_from_directory
import os
import librosa
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('my_model1.keras')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'aac', 'ogg', 'm4a', 'mp4', 'mov', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path):
    wav_file_path = input_path.rsplit('.', 1)[0] + '.wav'
    
    try:
        video_clip = VideoFileClip(input_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(wav_file_path, codec='pcm_s16le')
        audio_clip.close()
        video_clip.close()
        print(f"Converted to WAV: {wav_file_path}")
        return wav_file_path
    
    except Exception as e:
        print(f"Error in converting to WAV: {str(e)}")
        return None

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(f"Received file: {filename}, saved at {file_path}")

        # Convert file if needed
        converted_file_path = None
        if filename.lower().endswith(('.mp4', '.mov', '.avi')):
            converted_file_path = convert_to_wav(file_path)
            if not converted_file_path:
                return jsonify({'error': 'Error converting video file'}), 500
            audio_path = converted_file_path
        else:
            audio_path = file_path

        audio, sr = librosa.load(audio_path, sr=None)
        features = extract_features(audio, sr)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        emotion = decode_prediction(prediction)

        return jsonify({'prediction': emotion})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

    finally:
        if converted_file_path and os.path.exists(converted_file_path):
            os.remove(converted_file_path)
        if os.path.exists(file_path):
            os.remove(file_path)




def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr)) 

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) 

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) 

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) 

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) 

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spectral_centroid))

    return result



def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*2000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr = sampling_rate, n_steps = pitch_factor)

# shift time from left or right
def time_shift(data, sample_rate):
    shift_max = 0.2  # shift by 20% of the total duration
    shift = np.random.randint(int(sample_rate * shift_max))
    direction = np.random.choice(['left', 'right'])
    if direction == 'left':
        shift = -shift
    augmented_data = np.roll(data, shift)
    return augmented_data

# to reduce volume of loud sound
def dynamic_range_compression(data):
    compressor = np.random.uniform(0.5, 1.0)
    data = np.sign(data) * (1 - np.exp(-compressor * np.abs(data)))
    return data

# adjust difference of frequency components
def equalize(data, sr):
    eq = np.random.uniform(0.8, 1.2)
    return librosa.effects.preemphasis(data, coef=eq)

# simulate effect of sound
def reverb(data):
    reverb_effect = np.convolve(data, np.random.rand(1000), mode='same')
    return reverb_effect



def get_features(path):


    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))

    # data with time shifting
    shifted_data = time_shift(data, sample_rate)
    res4 = extract_features(shifted_data, sample_rate)
    result = np.vstack((result, res4))

    # data with dynamic range compression
    compressed_data = dynamic_range_compression(data)
    res5 = extract_features(compressed_data, sample_rate)
    result = np.vstack((result, res5))

    # data with equalization
    equalized_data = equalize(data, sample_rate)
    res6 = extract_features(equalized_data, sample_rate)
    result = np.vstack((result, res6))

    return result
def decode_prediction(prediction):
    emotions_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_index = np.argmax(prediction)
    predicted_emotion = emotions_labels[predicted_index]
    return predicted_emotion


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)