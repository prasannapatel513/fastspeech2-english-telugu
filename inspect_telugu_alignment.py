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

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCTC.from_pretrained(MODEL_NAME)
model.eval()

# Read transcript
df = pd.read_csv(os.path.join(DATASET, "manifest_22050.csv"))
row = df[df["audio_path"].str.endswith(AUDIO_NAME)]

if row.empty:
    raise ValueError("Audio not found in manifest.")

transcript = str(row.iloc[0]["transcript"]).strip().rstrip(".,!?;:")

# Read audio
audio_path = os.path.join(DATASET, "audio_22050", AUDIO_NAME)
audio, sr = sf.read(audio_path)

if audio.ndim == 2:
    audio = audio.mean(axis=1)

if sr != 16000:
    audio = scipy.signal.resample_poly(audio, 16000, sr)
    sr = 16000

# Prepare input
inputs = processor(
    audio,
    sampling_rate=sr,
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(inputs.input_values).logits

log_probs = torch.log_softmax(logits, dim=-1)

# Transcript tokens
token_ids = processor.tokenizer(
    transcript,
    add_special_tokens=False
).input_ids

if processor.tokenizer.unk_token_id in token_ids:
    raise ValueError("Unknown token found.")

targets = torch.tensor([token_ids], dtype=torch.int32)

# Forced alignment
aligned_tokens, scores = torchaudio.functional.forced_align(
    log_probs,
    targets,
    blank=processor.tokenizer.pad_token_id
)
# Compare model's own prediction with the dataset transcript
predicted_ids = torch.argmax(logits, dim=-1)
predicted_text = processor.batch_decode(predicted_ids)[0]

print("\nMODEL RECOGNITION:", predicted_text)
print("DATASET TRANSCRIPT:", transcript)

# Average alignment score for each character
print("\nCHARACTER ALIGNMENT SCORES")

for token_id in token_ids:
    positions = [
        i for i, value in enumerate(aligned_tokens[0].tolist())
        if value == token_id
    ]

    if positions:
        avg_score = scores[0, positions].mean().item()
        character = processor.tokenizer.convert_ids_to_tokens(
            [token_id]
        )[0]

        print(character, "Frames:", len(positions),
              "Average score:", round(avg_score, 4))

path = aligned_tokens[0].tolist()
blank_id = processor.tokenizer.pad_token_id

# Collapse consecutive repeated tokens and remove blanks
collapsed = []

for token_id in path:
    if not collapsed or token_id != collapsed[-1]:
        if token_id != blank_id:
            collapsed.append(token_id)

print("\nEXPECTED TOKEN IDS:", token_ids)
print("ALIGNED TOKEN IDS: ", collapsed)
print("SEQUENCE MATCH:", collapsed == token_ids)

# Extract timing for each token
frame_step = 320 / sr
duration = len(audio) / sr

print("\nCHARACTER TIMINGS")
print("-" * 55)

for token_id in token_ids:
    positions = [
        i for i, value in enumerate(path)
        if value == token_id
    ]

    if not positions:
        print("MISSING TOKEN:", token_id)
        continue

    start = positions[0] * frame_step
    end = (positions[-1] + 1) * frame_step
    end = min(end, duration)

    character = processor.tokenizer.convert_ids_to_tokens(
        [token_id]
    )[0]

    print(
        f"{character} | "
        f"Start: {start:.3f}s | "
        f"End: {end:.3f}s | "
        f"Duration: {end-start:.3f}s"
    )

print("\nAudio duration:", round(duration, 3), "seconds")
print("Alignment frames:", len(path))