# 📧 Spam Email Detector — CSC 309 AI Mini Project (Task 3)

**Federal University of Technology, Owerri**

## Overview
An AI-powered spam detector built with Python, Streamlit, and a pre-trained BERT model from Hugging Face.
No local training required — the model is downloaded automatically on first run.

## AI Concepts Covered
- Supervised Learning
- Text Classification
- Natural Language Processing (NLP)
- BERT Transformer (Transfer Learning)
- Binary Classification (Spam vs Ham)

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

> **Note:** On first launch, the BERT model (~17MB) will download automatically from Hugging Face.
> This requires an internet connection for the first run only. After that it is cached locally.

## Model Used
`mrm8488/bert-tiny-finetuned-sms-spam-detection`  
Pre-trained on the SMS Spam Collection Dataset (UCI Machine Learning Repository)

## File Structure
```
spam_detector/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
