# FastSpeech2 English–Telugu Text-to-Speech Implementation

## 1. Project Overview

### 1.1 Introduction

Text-to-Speech (TTS) is a technology that converts written text into spoken audio. This project focuses on exploring and implementing a multilingual Text-to-Speech system for English and Telugu using the FastSpeech2 architecture.

FastSpeech2 is a non-autoregressive neural TTS model that generates speech representations from input text. It uses duration, pitch, and energy information to improve speech generation.

### 1.2 Project Objectives

- Explore the FastSpeech2 architecture for English and Telugu speech synthesis.
- Collect and prepare English and Telugu speech datasets.
- Perform audio preprocessing and dataset validation.
- Prepare training and validation datasets.
- Explore text processing and speech alignment techniques.
- Evaluate the feasibility of multilingual speech synthesis.

### 1.3 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Implementation |
| PyTorch | Deep learning framework |
| FastSpeech2 | Text-to-Speech architecture |
| Hugging Face | Dataset and model access |
| Svarah | English speech dataset |
| IndicTTS Telugu | Telugu speech dataset |
| Librosa / SoundFile | Audio processing |
| NumPy / SciPy | Numerical and signal processing |
| VS Code | Development environment |
| Git / GitHub | Version control and project hosting |

### 1.4 Project Scope

The implementation covers dataset collection, audio conversion, transcript preparation, dataset splitting, preprocessing experiments, and investigation of speech alignment methods.

The English pipeline progressed through feature preprocessing. The Telugu pipeline progressed through dataset preparation and alignment feasibility testing. Full model training and end-to-end speech generation for both languages remain pending.
## 2. System Architecture

### 2.1 Overview

The proposed system follows a text-to-speech processing pipeline. It accepts written text, processes the input, extracts linguistic and acoustic features, and uses a speech synthesis model to generate audio.

### 2.2 Processing Pipeline

The planned processing pipeline consists of the following stages:

1. **Dataset Collection:** Collect English and Telugu speech recordings with their corresponding transcripts.
2. **Audio Preprocessing:** Convert audio files into a consistent format, including mono-channel audio and a sampling rate of 22,050 Hz.
3. **Text Processing:** Clean transcripts and prepare text representations suitable for the target language.
4. **Speech Alignment:** Determine the relationship between transcript units and audio frames to obtain duration information.
5. **Feature Extraction:** Extract mel-spectrogram, pitch, energy, and duration features.
6. **Model Training:** Train the FastSpeech2 model using prepared linguistic and acoustic features.
7. **Speech Generation:** Generate mel-spectrograms from input text and convert them into speech audio using a vocoder.

### 2.3 FastSpeech2 Architecture

FastSpeech2 is a non-autoregressive TTS architecture. Its major components are:

- **Input Embedding:** Converts text symbols into vector representations.
- **Encoder:** Learns contextual representations of the input text.
- **Variance Adaptor:** Uses duration, pitch, and energy information to control speech characteristics.
- **Decoder:** Generates mel-spectrogram representations.
- **Vocoder:** Converts mel-spectrograms into waveform audio.

### 2.4 Architecture Flow

```text
Input Text
    |
    v
Text Cleaning and Tokenization
    |
    v
Text Encoder
    |
    v
Variance Adaptor
(Duration, Pitch, Energy)
    |
    v
Mel-Spectrogram Decoder
    |
    v
Vocoder
    |
    v
Generated Speech Audio
```

### 2.5 Language-Specific Processing

**English:**
- Dataset: Svarah English speech.
- Text processing: English text cleaning and phoneme-based processing.
- Alignment: Montreal Forced Aligner (MFA).
- Acoustic features: Mel-spectrogram, pitch, energy, and duration.

**Telugu:**
- Dataset: IndicTTS Telugu.
- Text processing: Telugu Unicode transcript preparation.
- Audio processing: Resampling to 22,050 Hz.
- Alignment investigation: Telugu Wav2Vec2 CTC recognition and forced-alignment experiments.

The Telugu alignment experiment did not produce sufficiently reliable character-level timings. Therefore, Telugu duration feature generation and complete FastSpeech2 training have not been finalized.
## 3. English Dataset Preparation

### 3.1 Dataset Description

The Svarah dataset was used for the English speech-processing pipeline. It contains English speech recordings along with their corresponding text transcripts and speaker-related metadata.

The dataset was accessed through Hugging Face using the dataset repository `ai4bharat/Svarah`.

### 3.2 Dataset Statistics

| Description | Result |
|---|---:|
| Total original recordings | 6,656 |
| Total original audio duration | 9.61 hours |
| Successfully downloaded recordings | 6,656 |
| Audio and transcript pairs | 6,656 |

### 3.3 Audio and Transcript Preparation

The following operations were performed:

- Downloaded English speech recordings and associated transcripts.
- Organized audio and transcript files into separate folders.
- Prepared metadata containing filenames, durations, transcripts, and speaker information.
- Generated TextGrid alignment files using Montreal Forced Aligner (MFA).
- Reviewed out-of-vocabulary words and created a custom pronunciation dictionary.
- Re-ran alignment using the corrected pronunciation dictionary.
- Checked the alignment outputs before feature preprocessing.

### 3.4 Training and Validation Split

An expanded dataset split was prepared while avoiding recording overlap and ensuring that validation speakers were represented in the training data.

| Description | Recordings | Audio duration |
|---|---:|---:|
| Training candidates | 4,639 | 6.370 hours |
| Validation candidates | 564 | 1.408 hours |
| Total candidates | 5,203 | 7.778 hours |

### 3.5 Preprocessing Results

After preprocessing:

| Description | Recordings | Feature duration |
|---|---:|---:|
| Training | 4,507 | 4.7048 hours |
| Validation | 544 | 1.0041 hours |
| Successfully processed | 5,051 | 5.7089 hours |
| Skipped recordings | 152 | — |

The preprocessing pipeline generated acoustic features for the successfully processed recordings. The final training and validation feature durations were calculated from the generated mel-spectrogram frame counts.

### 3.6 Validation Checks

The following checks were performed:

- No overlapping recordings between training and validation.
- All validation speakers were represented in the training set.
- No missing mel-spectrogram features were found in the processed records.
- No frame-count and duration mismatches were found in the completed preprocessing checks.

### 3.7 English Dataset Conclusion

The English dataset was successfully downloaded, organized, aligned, and preprocessed for the FastSpeech2 pipeline.

The final processed training duration was approximately 4.70 hours, and the validation duration was approximately 1.00 hour. The training duration is slightly below the target of five hours.
## 4. Telugu Dataset Preparation

### 4.1 Dataset Description

The IndicTTS Telugu dataset was obtained from the Hugging Face repository:

`SPRINGLab/IndicTTS_Telugu`

The dataset contains Telugu speech recordings with corresponding text transcripts and gender information.

### 4.2 Dataset Statistics

| Description | Result |
|---|---:|
| Total recordings | 8,576 |
| Female recordings | 4,043 |
| Male recordings | 4,533 |
| Successfully downloaded | 8,576 |
| Failed downloads | 0 |
| Total measured audio duration | 36.71 hours |

The downloaded audio duration was measured from the actual WAV files. This differs from the approximate duration stated in the dataset information.

### 4.3 Audio Preprocessing

The following operations were completed:

- Downloaded all 8,576 Telugu speech recordings.
- Verified the availability of audio files and transcripts.
- Checked audio channels and sample rates.
- Converted all recordings to 22,050 Hz.
- Prepared a converted manifest and Telugu metadata file.
- Verified that all 8,576 recordings were converted successfully.

### 4.4 Training and Validation Split

A recording-level split was prepared using the target durations of approximately five hours for training and one hour for validation.

| Description | Recordings | Duration |
|---|---:|---:|
| Training | 1,178 | 5.013 hours |
| Validation | 237 | 1.021 hours |
| Unused recordings | 7,161 | 30.679 hours |
| Total | 8,576 | 36.713 hours |

Both male and female recordings were included in the training and validation subsets.

No recording overlap was found between the training and validation sets.

**Speaker limitation:** The dataset was organized using two gender-based speaker IDs. Therefore, this split prevents recording overlap but does not establish speaker-independent validation.

### 4.5 Telugu Text Recognition and Alignment Experiment

A Telugu Wav2Vec2 CTC model was investigated to explore transcript-to-audio alignment.

The model used was:

`anuragshas/wav2vec2-large-xlsr-53-telugu`

Initial testing showed that the model could recognize Telugu speech. A forced-alignment experiment was also performed on a sample recording.

However, further validation on ten recordings produced the following results:

| Evaluation | Result |
|---|---:|
| Recordings tested | 10 |
| Exact normalized recognition matches | 0 |
| Forced-alignment sequence matches |
## 5. Implementation Status

### 5.1 Completed Work

The following project components have been implemented:

1. FastSpeech2 project structure setup.
2. Python environment and required dependency setup.
3. English Svarah dataset downloading and organization.
4. English transcript and metadata preparation.
5. English speech alignment using MFA.
6. English custom pronunciation dictionary preparation.
7. English training and validation split.
8. English acoustic feature preprocessing.
9. Telugu IndicTTS dataset downloading.
10. Telugu audio conversion to 22,050 Hz.
11. Telugu transcript and metadata preparation.
12. Telugu training and validation split.
13. Telugu speech recognition and forced-alignment feasibility experiments.

### 5.2 Pending Work

The following components remain incomplete:

- Reliable Telugu character or phoneme alignment.
- Telugu duration, pitch, energy, and mel feature generation through the FastSpeech2 preprocessing pipeline.
- Complete Telugu FastSpeech2 model training.
- Complete English FastSpeech2 model training and evaluation.
- Final speech generation and quality evaluation for both languages.

## 6. Limitations and Challenges

### 6.1 Telugu Alignment Challenge

The available alignment experiment did not achieve reliable transcript-to-audio matching across the tested Telugu recordings. The resulting alignment sequences could not be used confidently to generate duration features.

### 6.2 Language-Specific Text Processing

Telugu uses a different writing system from English. The original FastSpeech2 text-processing configuration requires adaptation to support Telugu Unicode characters and appropriate linguistic units.

### 6.3 Computational Resources

The development environment used CPU-based PyTorch without CUDA acceleration. This limited the practicality of training large speech synthesis models within the available project timeline.

### 6.4 Time Constraints

Due to the limited implementation period, the project prioritized dataset preparation, preprocessing, validation, and technical feasibility experiments over full model training.

## 7. Future Enhancements

The following improvements are planned:

1. Develop a reliable Telugu-compatible alignment pipeline.
2. Extend the FastSpeech2 symbol inventory and text processing for Telugu.
3. Generate and validate Telugu acoustic features.
4. Train separate English and Telugu FastSpeech2 models.
5. Integrate a suitable vocoder for waveform generation.
6. Evaluate generated speech using objective metrics