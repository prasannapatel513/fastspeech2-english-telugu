# FastSpeech2 English–Telugu Text-to-Speech System

## 1. Project Overview

This project explores the implementation of a multilingual Text-to-Speech (TTS) system using the FastSpeech2 architecture for English and Telugu speech synthesis.

The objective is to prepare datasets, study the preprocessing and alignment pipeline, and investigate the requirements for generating natural speech from text.

## 2. Technologies Used

- Python
- PyTorch
- FastSpeech2
- SQL and data-processing utilities where applicable
- Librosa and SoundFile
- NumPy and SciPy
- Montreal Forced Aligner (MFA)
- Hugging Face datasets and models

## 3. Project Structure

```text
fastspeech2-english-telugu/
│
├── English/
│   ├── raw_data/
│   ├── preprocessed_data/
│   ├── output/
│   └── config/
│
├── Telugu/
│   ├── audio/
│   ├── checkpoints/
│   ├── config/
│   ├── model/
│   ├── preprocessor/
│   └── text/
│
├── Results/
│   ├── English/
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   └── Telugu/
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│
└── Report/
    └── Technical_Implementation_Report.md
```

## 4. Dataset Preparation

### English
- Dataset: AI4Bharat Svarah.
- Prepared audio and transcript metadata.
- Performed dataset splitting and preprocessing.
- Generated mel-spectrogram, duration, pitch, and energy features.
- Checked training and validation data separation.

### Telugu
- Dataset: AI4Bharat IndicTTS Telugu.
- Downloaded and inspected audio recordings and transcripts.
- Converted audio to 22,050 Hz.
- Prepared training and validation metadata.
- Checked recording overlap and dataset duration.

## 5. Implementation Status

| Component | Status |
|---|---|
| English dataset preparation | Completed |
| English preprocessing | Completed |
| English pretrained model inference test | Demonstrated |
| Telugu dataset preparation | Completed |
| Telugu audio conversion | Completed |
| Telugu alignment validation | Experimented; unsuccessful |
| English FastSpeech2 training | Pending |
| Telugu FastSpeech2 training | Pending |
| Final multilingual speech generation | Pending |

## 6. Limitations

- The available local environment uses CPU-only PyTorch.
- Telugu forced alignment did not produce reliable results in the validation experiment.
- Full acoustic model training and final multilingual evaluation remain incomplete.

## 7. Detailed Documentation

Refer to `Report/Technical_Implementation_Report.md` for the dataset statistics, implementation details, experiments, limitations, and future improvements.

## 8. Future Enhancements

- Complete reliable Telugu phoneme alignment.
- Generate duration, pitch, and energy features for Telugu.
- Train English and Telugu FastSpeech2 models.
- Integrate a compatible HiFi-GAN vocoder.
- Evaluate speech quality and intelligibility.
