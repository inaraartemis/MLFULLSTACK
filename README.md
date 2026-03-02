# 🔮 Luna AI: Lavender Edition

**Luna AI** is a premium, full-stack machine learning platform designed for high-performance predictive analytics and optical pattern recognition. Overhauled with a stunning **Lavender Design System**, it combines aesthetic excellence with powerful ML capabilities.

![Luna AI Dashboard](static/dashboard_preview.png) *(Preview placeholder)*

## ✨ Key Features

### 💜 Premium Aesthetic
- **Lavender Design System:** A cohesive UI/UX featuring glassmorphism, amethyst gradients, and modern "Outfit" typography.
- **Unified Dashboard:** A single-page application feel with a centralized command center.

### 🏥 Luna Predict: Health Suite
- **Engine:** Linear Regression
- **Dataset:** Diabetes Progression
- **Capability:** Predicts clinical disease progression metrics based on 10 physiological markers.

### ✍️ Luna Classify: Pattern Recognition
- **Engine:** Logistic Regression (LogReg-88)
- **Dataset:** UCI Handwritten Digits
- **Capability:** 
    - **Real-time Drawing:** Live canvas for digit sketching.
    - **External Input:** Support for image uploads and drag-and-drop analysis.
    - **Smart Pipeline:** Automatic inversion detection and 8x8 vectorization.

## 🚀 Quick Start

### 1. Requirements
Ensure you have Python 3.10+ installed.

### 2. Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/inaraartemis/MLFULLSTACK.git
cd MLFULLSTACK
pip install -r requirements.txt
```

### 3. Training the Models
If the models are not present, run the training scripts:
```bash
python train_regression.py
python train_classification.py
```

### 4. Run the Platform
Launch the FastAPI server:
```bash
uvicorn app:app --reload
```
Navigate to `http://127.0.0.1:8000` in your browser.

---
**Made by Arpita Mahapatra • 2026**
