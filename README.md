# Speech Technology Assignment 2

This project is my implementation for Assignment 2, focusing on building a speech-to-speech pipeline using PyTorch and Transformers. The goal was to take a noisy classroom lecture, transcribe it accurately with technical terms, and then synthesize a version of it in Marathi.

## Project Features
1. **Denoising:** I used stationary noise reduction to clean up background noise.
2. **Biased Transcription:** I used Whisper-medium and injected specific logit biases so that the model wouldn't mess up technical terms like "Stochastic" and "Cepstrum."
3. **Marathi Synthesis:** I used the Facebook MMS (Massively Multilingual Speech) model for zero-shot TTS to convert the text into Marathi.

## Files in this Repo
- `Speech_Assignment2.ipynb`: The full code for the pipeline.
- `Report.pdf`: Technical report in IEEE format.
- `Implementation_Note.pdf`: A 1-page note on my design choices.
- `outputs/`: Folder containing the denoised and synthesized audio files.

## How to Run
1. Open the notebook in Google Colab (make sure GPU is on).
2. Install dependencies: `!pip install transformers torch torchaudio librosa noisereduce`
3. Upload your `extracted_sample.wav` and run the cells.
