import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

# Extract MFCC features from audio
def extract_mfcc(file_path, max_length=216):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < max_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load RAVDESS dataset
def load_ravdess(dataset_path):
    X, y = [], []
    for actor_folder in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_folder)
        if os.path.isdir(actor_path):
            for file_name in os.listdir(actor_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(actor_path, file_name)
                    # Extract emotion from filename (e.g., 03-01-01-01-01-01-01.wav, where 01 is emotion)
                    emotion = int(file_name.split('-')[2]) - 1  # RAVDESS emotions are 1-indexed
                    if emotion < len(emotion_labels):  # Ensure valid emotion
                        mfcc = extract_mfcc(file_path)
                        if mfcc is not None:
                            X.append(mfcc)
                            y.append(emotion)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Preprocess audio data
def preprocess_audio(X, y):
    X = X[..., np.newaxis]  # Add channel dimension for Conv1D
    le = LabelEncoder()
    y = tf.keras.utils.to_categorical(le.fit_transform(y), num_classes=len(emotion_labels))
    return X, y

# Build CNN model for audio
def build_audio_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(emotion_labels), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    audio_dataset_path = os.path.join(os.getcwd(), 'RAVDESS')  # Update with your dataset path
    audio_model_path = 'audio_emotion_recognition_model.h5'

    if os.path.exists(audio_dataset_path):
        print("🔄 Training audio model...")
        X, y = load_ravdess(audio_dataset_path)
        X, y = preprocess_audio(X, y)
        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y, test_size=0.2, random_state=42)
        audio_model = build_audio_model(input_shape=(X_train_a.shape[1], X_train_a.shape[2]))
        history = audio_model.fit(X_train_a, y_train_a, batch_size=32, epochs=50,
                                  validation_data=(X_test_a, y_test_a), verbose=1)
        audio_model.save(audio_model_path)
        print(f"Model saved as {audio_model_path}")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')

        plt.savefig('audio_training_history.png')
        print("Training history plot saved as audio_training_history.png")
    else:
        print("❌ RAVDESS dataset not found. Please download from https://zenodo.org/record/1188976")

if __name__ == '__main__':
    main()