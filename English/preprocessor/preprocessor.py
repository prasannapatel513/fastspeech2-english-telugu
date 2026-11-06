import os
import random
import json
import csv
import re

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        if self.config.get("dataset") == "Svarah":
            return self.build_svarah_from_path()
        
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if not wav_name.endswith(".wav"):
                    continue

                basename = wav_name.split(".")[0]

                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )

                if not os.path.exists(tg_path):
                    continue

                ret = self.process_utterance(speaker, basename)

                if ret is None:
                    continue

                info, pitch, energy, n = ret

                out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))

                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out
    def build_svarah_from_path(self):
        # Create feature folders
        for folder in ["mel", "pitch", "energy", "duration"]:
            os.makedirs(os.path.join(self.out_dir, folder), exist_ok=True)

        train_csv = self.config["path"]["train_csv"]
        val_csv = self.config["path"]["val_csv"]

        # Read the verified CSV files
        def read_split(csv_path):
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(
                    "CSV file not found: {}".format(csv_path)
                )

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))

        train_rows = read_split(train_csv)
        val_rows = read_split(val_csv)

        # Get filenames and speaker IDs
        def prepare_rows(rows, split_name):
            prepared = []

            for row in rows:
                filename = row["filename"]
                basename = os.path.splitext(filename)[0]

                match = re.match(r"^\d+_(f\d+)_chunk_", basename)

                if not match:
                    raise ValueError(
                        "Cannot identify speaker from filename: {}".format(
                            filename
                        )
                    )

                speaker = match.group(1)

                wav_path = os.path.join(
                    self.in_dir, speaker, basename + ".wav"
                )
                lab_path = os.path.join(
                    self.in_dir, speaker, basename + ".lab"
                )
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker,
                    basename + ".TextGrid"
                )

                for path in [wav_path, lab_path, tg_path]:
                    if not os.path.isfile(path):
                        raise FileNotFoundError(
                            "Required file missing: {}".format(path)
                        )

                prepared.append({
                    "speaker": speaker,
                    "basename": basename,
                    "filename": filename,
                    "split": split_name
                })

            return prepared

        train_items = prepare_rows(train_rows, "train")
        val_items = prepare_rows(val_rows, "val")

        # Prevent train-validation overlap
        train_names = {item["filename"] for item in train_items}
        val_names = {item["filename"] for item in val_items}

        if train_names.intersection(val_names):
            raise ValueError("Training and validation files overlap!")

        # Every validation speaker must exist in training
        train_speakers = sorted({
            item["speaker"] for item in train_items
        })

        val_speakers = {
            item["speaker"] for item in val_items
        }

        unknown_speakers = val_speakers - set(train_speakers)

        if unknown_speakers:
            raise ValueError(
                "Validation contains speakers absent from training: {}".format(
                    sorted(unknown_speakers)
                )
            )

        print("Training recordings:", len(train_items))
        print("Validation recordings:", len(val_items))
        print("Training speakers in CSV:", len(train_speakers))

        # Process recordings
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        processed_train = []
        processed_val = []

        skipped = []
        total_frames = 0

        all_items = train_items + val_items

        for item in tqdm(all_items, desc="Processing Svarah"):
            speaker = item["speaker"]
            basename = item["basename"]

            ret = self.process_utterance(speaker, basename)

            if ret is None:
                skipped.append(item["filename"])
                continue

            info, pitch, energy, n_frames = ret

            result = {
                "info": info,
                "speaker": speaker,
                "basename": basename,
                "split": item["split"]
            }

            if item["split"] == "train":
                processed_train.append(result)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(
                        pitch.reshape((-1, 1))
                    )

                if len(energy) > 0:
                    energy_scaler.partial_fit(
                        energy.reshape((-1, 1))
                    )

            else:
                processed_val.append(result)

            total_frames += n_frames
                # Include only speakers with successfully processed training data
        processed_train_speakers = sorted({
            item["speaker"] for item in processed_train
        })

        processed_val_speakers = {
            item["speaker"] for item in processed_val
        }

        unknown_speakers = (
            processed_val_speakers - set(processed_train_speakers)
        )

        if unknown_speakers:
            raise RuntimeError(
                "These validation speakers have no successful training "
                "recordings: {}".format(sorted(unknown_speakers))
            )

        speakers = {
            speaker: index
            for index, speaker in enumerate(processed_train_speakers)
        }

        print("Successfully processed training speakers:", len(speakers))

        # Ensure training features were generated
        if not processed_train:
            raise RuntimeError(
                "No training recordings were successfully processed."
            )

        if not hasattr(pitch_scaler, "mean_"):
            raise RuntimeError("No usable training pitch values found.")

        if not hasattr(energy_scaler, "mean_"):
            raise RuntimeError("No usable training energy values found.")

        # Calculate normalization statistics from training only
        if self.pitch_normalization:
            pitch_mean = float(pitch_scaler.mean_[0])
            pitch_std = float(pitch_scaler.scale_[0])
        else:
            pitch_mean = 0.0
            pitch_std = 1.0

        if self.energy_normalization:
            energy_mean = float(energy_scaler.mean_[0])
            energy_std = float(energy_scaler.scale_[0])
        else:
            energy_mean = 0.0
            energy_std = 1.0

        # Normalize only files generated in this run
        # Min/max statistics are calculated from training files only
        def normalize_features(feature, mean, std):
            train_min = np.inf
            train_max = -np.inf

            for item in processed_train + processed_val:
                speaker = item["speaker"]
                basename = item["basename"]

                filename = "{}-{}-{}.npy".format(
                    speaker, feature, basename
                )

                path = os.path.join(
                    self.out_dir, feature, filename
                )

                values = np.load(path)
                values = (values - mean) / std
                np.save(path, values)

                if item["split"] == "train" and values.size > 0:
                    train_min = min(train_min, float(np.min(values)))
                    train_max = max(train_max, float(np.max(values)))

            if not np.isfinite(train_min) or not np.isfinite(train_max):
                raise RuntimeError(
                    "Could not calculate {} training range.".format(feature)
                )

            return train_min, train_max

        pitch_min, pitch_max = normalize_features(
            "pitch", pitch_mean, pitch_std
        )

        energy_min, energy_max = normalize_features(
            "energy", energy_mean, energy_std
        )

        # Save speaker mapping
        with open(
            os.path.join(self.out_dir, "speakers.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(speakers, f, indent=2)

        # Save normalization statistics
        stats = {
            "pitch": [
                float(pitch_min),
                float(pitch_max),
                float(pitch_mean),
                float(pitch_std)
            ],
            "energy": [
                float(energy_min),
                float(energy_max),
                float(energy_mean),
                float(energy_std)
            ]
        }

        with open(
            os.path.join(self.out_dir, "stats.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(stats, f, indent=2)

        # Write the exact training and validation metadata
        train_metadata = [
            item["info"] for item in processed_train
        ]

        val_metadata = [
            item["info"] for item in processed_val
        ]

        with open(
            os.path.join(self.out_dir, "train.txt"),
            "w",
            encoding="utf-8"
        ) as f:
            for line in train_metadata:
                f.write(line + "\n")

        with open(
            os.path.join(self.out_dir, "val.txt"),
            "w",
            encoding="utf-8"
        ) as f:
            for line in val_metadata:
                f.write(line + "\n")

        print("\nSvarah preprocessing completed.")
        print("Successful training recordings:", len(processed_train))
        print("Successful validation recordings:", len(processed_val))
        print("Skipped recordings:", len(skipped))
        print(
            "Processed audio duration: {:.4f} hours".format(
                total_frames * self.hop_length /
                self.sampling_rate / 3600
            )
        )

        if skipped:
            print("Skipped filenames (first 20):")
            for filename in skipped[:20]:
                print(filename)

        return train_metadata + val_metadata

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None
        if basename == "LJ001-0001":
            print("DEBUG start:", start)
            print("DEBUG end:", end)
            print("DEBUG phone count:", len(phone))

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        if basename == "LJ001-0001":
           print("DEBUG audio length:", len(wav))
           print("DEBUG pitch frames:", len(pitch))
           print("DEBUG voiced frames:", np.sum(pitch != 0))
           print("DEBUG maximum pitch:", np.max(pitch))

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
