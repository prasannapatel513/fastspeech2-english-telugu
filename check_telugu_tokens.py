import os
import pandas as pd
from transformers import AutoProcessor

MODEL_NAME = "anuragshas/wav2vec2-large-xlsr-53-telugu"

DATASET = (
    r"C:\Users\Laxmi prasanna\OneDrive\Documents"
    r"\Telugu_TTS_Project\datasets\IndicTTS_Telugu_HF"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

df = pd.read_csv(os.path.join(DATASET, "manifest_22050.csv"))

audio_file = "train_telugufemale_00001.wav"
row = df[df["audio_path"].str.endswith(audio_file)]

transcript = str(row.iloc[0]["transcript"])

# Remove punctuation only
clean_text = transcript.strip().rstrip(".,!?;:")

token_ids = processor.tokenizer(
    clean_text,
    add_special_tokens=False
).input_ids

tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)

print("Original:", transcript)
print("Cleaned:", clean_text)
print("Tokens:", tokens)
print("Unknown token ID:", processor.tokenizer.unk_token_id)
print("Contains unknown token:", processor.tokenizer.unk_token_id in token_ids)
print("Decoded:", processor.tokenizer.decode(token_ids))