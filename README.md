<h1 style="text-align: center;">Audio Emotion Recognition</h1>


https://github.com/user-attachments/assets/8e58168b-915d-4463-ace3-20a77ba42b4e



## Project Overview

This project aims to develop a model for recognizing emotions from audio recordings. Using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks, the model classifies emotions based on audio features extracted from multiple datasets.

## Datasets Used

- **RAVDESS:** Includes audio and video recordings with emotional speech.
- **CREMA-D:** Contains emotional speech recordings from different actors.
- **TESS:** Toronto Emotional Speech Set with various emotions.
- **SAVEE:** Surrey Audio-Visual Expressed Emotion dataset.

## Libraries and Tools

- **Python Libraries:**
  - `pandas` and `numpy` for data handling
  - `librosa` for audio processing
  - `seaborn` and `matplotlib` for data visualization
  - `scikit-learn` for preprocessing and metrics
  - `keras` and `tensorflow` for building and training the model
- **Google Colab:** For initial dataset exploration and augmentation
- **Flask:** For deploying the model through a web interface

## Getting Started

### Prerequisites

- **Python:** Make sure Python 3.x is installed on your system.
# Settin up virtual environment

    python -m venv venv 
    source venv/bin/activate  # For Unix/macOS
    venv\Scripts\activate  # For Windows 
    pip install -r requirements.txt
    python app.py
## Project Structure
- `app.py`: The Flask application handling file uploads and predictions.
- `model1.keras`: The trained model file.
- `data_path.csv`: DataFrame with paths and emotions for audio files.
- `requirements.txt`: List of Python packages required for the project.
- `notebooks`: Jupyter notebooks for data exploration and model training.

## Project Background
 I originally took the initial code for data loading and model building from a kaggle notebook [here](https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition) . After that I made changes in the preprocessing, feature extraction, and changed the architecture of model. Previously the accruacy of the model was `61%`.
 The architecture that I made has almost `70%` accuracy. Still I think improvements can be done to this. But I did not have enough GPU resources to test it with multiple augmentations and feature extraction.


