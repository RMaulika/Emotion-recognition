## Emotion Recognition System using Facial and Audio Inputs

This project is a hybrid **Emotion Recognition System** that detects human emotions using both facial expressions (from images) and voice data (from audio clips). It combines deep learning techniques to achieve accurate results for real-time applications.

-----------------------------------------

##  Project Structure

EmotionRecognitionProject/
├── RAVDESS/ # Audio dataset folder
├── fer2013.csv # CSV file for facial emotion dataset
├── train_audio_emotion_model.py # Training script for audio emotion model
├── train_facial_emotion_model.py # Training script for facial emotion model
├── emotion_recognition.py # Main script to run emotion recognition
├── audio_emotion_recognition_model.h5
├── emotion_recognition_model.h5 # Trained facial model
├── audio_training_history.png # Plot of audio model training
├── facial_training_history.png # Plot of facial model training
├── requirements.txt

-----------------------------------------

##  Download Required Files

Some files are too large to upload directly to GitHub. You can download them from Google Drive and place them in the project folder.

-  [RAVDESS Audio Dataset](https://drive.google.com/drive/folders/1Hxq_blmHjSOcylV-ELaA_YtD8UDvNqcy?usp=drive_link)
-  [FER-2013 CSV Dataset](https://drive.google.com/file/d/1hHfIl5PFYmac9iMULE3jAbclE_YcWabe/view?usp=drive_link))
-  [Facial Emotion Model (emotion_recognition_model.h5)](https://drive.google.com/file/d/1eyCUsg4sOqxGQDYpXrEaoDrIiFI4-sHC/view?usp=drive_link)

>  After downloading, place the files in the root folder as shown in the project structure.

-----------------------------------------

##  Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/RMaulika/Emotion-recognition.git
cd Emotion-recognition
```
2. **Create a Virtual Environment and Install Dependencies**

python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

-----------------------------------------

## **How to Run the Project**

-> Make sure all model files and datasets are in place.

To train or use facial emotion recognition:
python train_facial_emotion_model.py     # Train facial model
python emotion_recognition.py            # Run predictions

To train audio emotion recognition:
python train_audio_emotion_model.py      # Train audio model

-----------------------------------------

## **Requirements**

Dependencies are listed in requirements.txt. The major ones include:

->tensorflow
->numpy
->opencv-python
->librosa
->pyaudio
->pandas
->matplotlib
->scikit-learn

-----------------------------------------

## **Emotions Detected**
->Angry 
->Happy 😄
->Sad 😢
->Neutral 😐
->Surprise 😲
->Fear 😨
->Disgust 🤢
->Calm 😌 (in audio)

-----------------------------------------

## **Output Visualization**
Training history plots for both models:

facial_training_history.png
audio_training_history.png

-----------------------------------------

## **Notes**
GitHub has a 25MB file size limit. Large files are provided via Google Drive links.

Allow microphone access if using real-time audio input.

FER-2013 and RAVDESS datasets are required for training.

