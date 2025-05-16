Multilingual Emotion Detection in Voice
This project performs multilingual speech transcription and emotion detection from audio in video files. It combines OpenAI Whisper for speech recognition and machine learning techniques for audio-based emotion classification.

🔍 Features
🎤 Speech-to-Text Transcription using Whisper

🌍 Multilingual Support

😄 Emotion Detection using MFCC and pitch features

🎥 Video-to-Audio Conversion (MP4 to WAV)

🧠 SVM Classifier for emotion recognition

📦 Dependencies
Install the required packages:

bash
Copy
Edit
pip install openai-whisper librosa scikit-learn moviepy
🛠️ How It Works
Convert Video to Audio
MP4 files are converted to WAV using moviepy.

Transcribe Audio
The Whisper model generates the transcription and detects the spoken language.

Extract Audio Features
Features like MFCCs and pitch are extracted using librosa.

Emotion Classification
A simple SVM classifier is used to predict emotions such as happy, sad, angry, and neutral.

🚀 Quick Start
python
Copy
Edit
# Load model and train classifier
model = whisper.load_model("base")
emotion_classifier, label_encoder = train_emotion_classifier()

# Run the full pipeline
emotion_aware_speech_recognition("your_video.mp4")
📁 File Structure
transcribe_audio(): Transcribes audio and returns language.

extract_audio_features(): Extracts MFCC and pitch.

train_emotion_classifier(): Trains a dummy SVM on synthetic data.

classify_emotion(): Predicts emotion based on audio features.

convert_mp4_to_wav(): Converts video to WAV.

emotion_aware_speech_recognition(): Full pipeline.

🧪 Notes
The classifier is currently trained on random synthetic data. For real-world applications, you should train it with labeled emotional speech datasets like RAVDESS or EMO-DB.

📜 License
This project is open-source and intended for educational and research purposes.

Would you like this saved as a README.md file in your project? 








