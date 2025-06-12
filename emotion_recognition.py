#.\.venv\Scripts\activate
#deactivate
import numpy as np
import pandas as pd
import cv2
import librosa
import pyaudio
import wave
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import time
import uuid

# Common emotion labels (aligned between FER-2013 and RAVDESS)
emotion_labels = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

# --- Facial Emotion Recognition ---
# Load FER-2013 dataset
def load_fer2013(dataset_path):
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split()]
        face = np.array(face).reshape(width, height)
        faces.append(face)
    faces = np.array(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions

# Preprocess facial images
def preprocess_faces(images, emotions):
    images = images.astype('float32') / 255.0
    return images, emotions

# Build facial CNN model
def build_facial_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(emotion_labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data augmentation for images
def create_image_datagen():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# --- Audio Emotion Recognition ---
# Extract MFCC features
def extract_mfcc(file_path=None, audio_data=None, sr=22050, max_length=216):
    try:
        if file_path:
            audio, sr = librosa.load(file_path, sr=sr)
        else:
            audio = audio_data
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < max_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]
        return mfcc
    except Exception as e:
        print(f"Error processing audio: {e}")
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
                    emotion = int(file_name.split('-')[2]) - 1
                    if emotion < len(emotion_labels):
                        mfcc = extract_mfcc(file_path)
                        if mfcc is not None:
                            X.append(mfcc)
                            y.append(emotion)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Preprocess audio data
def preprocess_audio(X, y):
    X = X[..., np.newaxis]
    le = LabelEncoder()
    y = tf.keras.utils.to_categorical(le.fit_transform(y), num_classes=len(emotion_labels))
    return X, y

# Build audio CNN model
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

# --- Real-Time Processing ---
# Preprocess frame for facial model
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (48, 48))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=[0, -1])
    return frame

# Record audio for a short duration
def record_audio(duration=3, rate=22050, channels=1, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.float32))
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.concatenate(frames)

# Combine predictions
def combine_predictions(face_pred, audio_pred, face_weight=0.6):
    if audio_pred is None:
        return face_pred
    return face_weight * face_pred + (1 - face_weight) * audio_pred

# Main function
def main():
    # Train or load facial model
    face_dataset_path = 'fer2013.csv'
    face_model_path = 'emotion_recognition_model.h5'
    if os.path.exists(face_model_path):
        print(f"✅ Found existing facial model at {face_model_path}, skipping training.")
        face_model = tf.keras.models.load_model(face_model_path)
    elif os.path.exists(face_dataset_path):
        print("🔄 Training facial model...")
        images, emotions = load_fer2013(face_dataset_path)
        images, emotions = preprocess_faces(images, emotions)
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(images, emotions, test_size=0.2, random_state=42)
        face_model = build_facial_model()
        datagen = create_image_datagen()
        datagen.fit(X_train_f)
        face_history = face_model.fit(datagen.flow(X_train_f, y_train_f, batch_size=64),
                                    epochs=30, validation_data=(X_test_f, y_test_f), verbose=1)
        face_model.save(face_model_path)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(face_history.history['accuracy'], label='Training Accuracy')
        plt.plot(face_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Facial Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(face_history.history['loss'], label='Training Loss')
        plt.plot(face_history.history['val_loss'], label='Validation Loss')
        plt.title('Facial Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('facial_training_history.png')
    else:
        print("❌ FER-2013 dataset not found. Skipping facial model training.")
        face_model = None

    # Train or load audio model
    audio_dataset_path = 'RAVDESS'
    audio_model_path = 'audio_emotion_recognition_model.h5'
    if os.path.exists(audio_model_path):
        print(f"✅ Found existing audio model at {audio_model_path}, skipping training.")
        audio_model = tf.keras.models.load_model(audio_model_path)
    elif os.path.exists(audio_dataset_path):
        print("🔄 Training audio model...")
        X, y = load_ravdess(audio_dataset_path)
        X, y = preprocess_audio(X, y)
        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y, test_size=0.2, random_state=42)
        audio_model = build_audio_model(input_shape=(X_train_a.shape[1], X_train_a.shape[2]))
        audio_history = audio_model.fit(X_train_a, y_train_a, batch_size=32, epochs=50, validation_data=(X_test_a, y_test_a), verbose=1)
        audio_model.save(audio_model_path)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(audio_history.history['accuracy'], label='Training Accuracy')
        plt.plot(audio_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Audio Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(audio_history.history['loss'], label='Training Loss')
        plt.plot(audio_history.history['val_loss'], label='Validation Loss')
        plt.title('Audio Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('audio_training_history.png')
    else:
        print("❌ RAVDESS dataset not found. Skipping audio model training.")
        audio_model = None


    # Real-time webcam and audio processing
    face_model = tf.keras.models.load_model('emotion_recognition_model.h5')
    audio_model = tf.keras.models.load_model('audio_emotion_recognition_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    last_audio_time = time.time()
    audio_interval = 3  # Process audio every 3 seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Facial emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_pred = None
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            processed_face = preprocess_frame(face)
            face_pred = face_model.predict(processed_face)[0]
            face_emotion = emotion_labels[np.argmax(face_pred)]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {face_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Audio emotion detection
        audio_pred = None
        if time.time() - last_audio_time >= audio_interval:
            audio_data = record_audio(duration=3)
            mfcc = extract_mfcc(audio_data=audio_data)
            if mfcc is not None:
                mfcc = mfcc[np.newaxis, ..., np.newaxis]
                audio_pred = audio_model.predict(mfcc)[0]
                audio_emotion = emotion_labels[np.argmax(audio_pred)]
                last_audio_time = time.time()
            else:
                audio_emotion = "Audio Error"
        else:
            audio_emotion = "Processing..."
        
        # Combined prediction
        if face_pred is not None and audio_pred is not None:
            combined_pred = combine_predictions(face_pred, audio_pred)
            combined_emotion = emotion_labels[np.argmax(combined_pred)]
        else:
            combined_emotion = face_emotion if face_pred is not None else "No Face"
        
        # Display results
        cv2.putText(frame, f"Audio: {audio_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Combined: {combined_emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow('Multimodal Emotion Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()