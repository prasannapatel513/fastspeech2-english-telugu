import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForCTC

MODEL_NAME = "anuragshas/wav2vec2-large-xlsr-53-telugu"

audio_path = (
    r"C:\Users\Laxmi prasanna\OneDrive\Documents"
    r"\Telugu_TTS_Project\datasets\IndicTTS_Telugu_HF"
    r"\audio_22050"
)

# Load the model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCTC.from_pretrained(MODEL_NAME)
model.eval()

# Read one WAV file
import os
wav_files = [f for f in os.listdir(audio_path) if f.endswith(".wav")]
wav_file = os.path.join(audio_path, wav_files[0])

audio, sample_rate = sf.read(wav_file)

# Convert stereo to mono if needed
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Resample to 16 kHz if needed
import scipy.signal
if sample_rate != 16000:
    audio = scipy.signal.resample_poly(
        audio, 16000, sample_rate
    )
    sample_rate = 16000

# Run speech recognition
inputs = processor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    logits = model(inputs.input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print("MODEL TEST COMPLETED")
print("Audio file:", wav_files[0])
print("Recognized Telugu:", transcription)