import os
import torch
import soundfile as sf
import scipy.signal
import pandas as pd
import torchaudio

from transformers import AutoProcessor, AutoModelForCTC

MODEL_NAME = "anuragshas/wav2vec2-large-xlsr-53-telugu"

DATASET = (
    r"C:\Users\Laxmi prasanna\OneDrive\Documents"
    r"\Telugu_TTS_Project\datasets\IndicTTS_Telugu_HF"
)

AUDIO_NAME = "train_telugufemale_00001.wav"

# 1. Load model and processor
print("Loading Telugu model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCTC.from_pretrained(MODEL_NAME)
model.eval()

# 2. Read transcript
df = pd.read_csv(os.path.join(DATASET, "manifest_22050.csv"))

row = df[df["audio_path"].str.endswith(AUDIO_NAME)]

if row.empty:
    raise ValueError("Audio file not found in manifest.")

transcript = str(row.iloc[0]["transcript"]).strip().rstrip(".,!?;:")

# 3. Read audio
audio_path = os.path.join(
    DATASET, "audio_22050", AUDIO_NAME
)

audio, sr = sf.read(audio_path)

if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Resample to 16 kHz
if sr != 16000:
    audio = scipy.signal.resample_poly(audio, 16000, sr)
    sr = 16000

# 4. Convert audio to model input
inputs = processor(
    audio,
    sampling_rate=sr,
    return_tensors="pt"
)

# 5. Get model logits
print("Generating model logits...")

with torch.no_grad():
    logits = model(inputs.input_values).logits

log_probs = torch.log_softmax(logits, dim=-1)

# 6. Convert transcript into token IDs
token_ids = processor.tokenizer(
    transcript,
    add_special_tokens=False
).input_ids

if processor.tokenizer.unk_token_id in token_ids:
    raise ValueError("Transcript contains unsupported tokens.")

# 7. Perform forced alignment
targets = torch.tensor([token_ids], dtype=torch.int32)

print("Running forced alignment...")

result = torchaudio.functional.forced_align(
    log_probs,
    targets,
    blank=processor.tokenizer.pad_token_id
)

aligned_tokens, scores = result

# 8. Display results
print("\nFORCED ALIGNMENT COMPLETED")
print("Transcript:", transcript)
print("Token count:", len(token_ids))
print("Aligned token count:", aligned_tokens.shape[1])
print("Audio duration:", round(len(audio) / sr, 2), "seconds")

print("\nAligned token IDs:")
print(aligned_tokens[0].tolist())

print("\nAlignment scores:")
print(scores[0].tolist())