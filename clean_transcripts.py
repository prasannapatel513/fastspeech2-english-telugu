
import glob
import re
import os

# Find all Svarah transcript files
files = glob.glob("raw_data/Svarah/*/*.lab")

print("Total transcript files found:", len(files))

for p in files:

    # Read transcript
    with open(p, "r", encoding="utf-8") as f:
        t = f.read().lower()

    # Remove tags such as [noise] and <tag>
    t = re.sub(r"\[.*?\]|<.*?>", " ", t)

    # Keep only English letters, apostrophes and spaces
    t = re.sub(r"[^a-z' ]", " ", t)

    # Remove extra spaces
    t = re.sub(r"\s+", " ", t).strip()

    # Save cleaned transcript
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)

    # Check whether transcript is empty
    if not t:
        wav_file = os.path.splitext(p)[0] + ".wav"

        print("Empty transcript:", p)

        os.remove(p)

        if os.path.exists(wav_file):
            os.remove(wav_file)
            print("Deleted matching audio:", wav_file)

print("Transcript cleaning completed!")
