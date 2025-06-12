import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

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
    faces = np.expand_dims(faces, -1)  # Add channel dimension
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions

# Preprocess images
def preprocess_data(images, emotions):
    images = images.astype('float32') / 255.0  # Normalize pixel values
    return images, emotions

# Build CNN model
def build_model():
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

# Data augmentation
def create_data_generator():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# Main function
def main():
    # Load and preprocess dataset
    dataset_path = 'fer2013.csv'  # Update with your dataset path
    if not os.path.exists(dataset_path):
        print("Error: fer2013.csv not found. Please download from https://www.kaggle.com/datasets/msambare/fer2013")
        return
    
    images, emotions = load_fer2013(dataset_path)
    images, emotions = preprocess_data(images, emotions)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(images, emotions, test_size=0.2, random_state=42)
    
    # Data augmentation
    datagen = create_data_generator()
    datagen.fit(X_train)
    
    # Build and train model
    model = build_model()
    history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                        epochs=30,
                        validation_data=(X_test, y_test),
                        verbose=1)
    
    # Save model
    model.save('emotion_recognition_model.h5')
    print("Model saved as emotion_recognition_model.h5")
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('facial_training_history.png')
    print("Training history plot saved as facial_training_history.png")

if __name__ == '__main__':
    main()