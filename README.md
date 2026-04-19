# Code-Switched Speech Pipeline & Zero-Shot Voice Cloning

**Speech Understanding – Assignment 2**  
**Author:** Ajinkya Ghodake (M25DE1035)  

---

## 📌 Project Overview

This repository implements an end-to-end PyTorch-based speech processing pipeline for **Code-Switched (Hinglish) academic lectures**, followed by **zero-shot cross-lingual speech synthesis in Marathi**.

The system integrates:
- Speech-to-Text (STT)
- Language Identification (LID)
- Phonetic normalization
- Prosody alignment using Dynamic Time Warping (DTW)
- Zero-shot TTS synthesis
- Adversarial robustness evaluation using FGSM

The design emphasizes **custom differentiable components** to maintain gradient flow and enable adversarial testing.

---

## 🚀 Key Features & Architecture

### 🔹 Task 1: Code-Switched STT & LID
- Frame-level Multi-Head LID using Wav2Vec2 latent representations
- Classification of English vs Hindi speech segments
- Target performance: **F1-score > 0.85**
- Noise reduction using `noisereduce`
- Constrained decoding using `SequenceBiasLogitsProcessor` (Whisper)

---

### 🔹 Task 2: Phonetic Mapping
- Deterministic mapping of Hinglish OOV words
- Conversion into IPA-inspired phonetic representations
- Enables consistent downstream synthesis

---

### 🔹 Task 3: Zero-Shot TTS & Prosody Warping
- Marathi synthesis using Meta MMS-TTS (`facebook/mms-tts-mar`)
- Dynamic Time Warping (DTW) applied over:
  - $F_0$ (fundamental frequency)
  - RMS energy contours
- Prosody alignment with reference speaker style
- Target metric: **MCD < 8.0**

---

### 🔹 Task 4: Adversarial Robustness & Anti-Spoofing
- LFCC + CNN-based deepfake detection system
- Target EER: **< 10%**
- FGSM-based adversarial attack generation
- Strict inaudibility constraint:
  - SNR > 40 dB
- Differentiable preprocessing ensures gradient flow

---

## 📂 Repository Structure

```text
M25DE1035_PA1/
│
├── input/
│   ├── original_segment.wav        # 10-min Hinglish lecture audio
│   └── student_voice_ref.wav      # Reference voice for cloning
│
├── assignment_outputs/
│   ├── temp_flat.wav              # Baseline (non-warped synthesis)
│   ├── denoised_input.wav         # Cleaned audio
│   └── output_LRL_cloned.wav      # Final Marathi synthesized output
│
├── mtech_speech_pipeline.py      # End-to-end pipeline script
├── M25DE1035_Report.pdf          # IEEE-style technical report
└── README.md
