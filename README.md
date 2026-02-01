
# RehabSense – AI Rehabilitation Monitoring Platform

RehabSense is an AI-powered rehabilitation and remote health monitoring system that analyzes physiological and behavioral health data using machine learning models. The platform supports real-time inference, analytics, and progress tracking through a scalable backend and interactive frontend.

---

## Project Structure

```
RehabSense/
├── backend/
│   └── app.py                  # Flask backend API
├── data/
│   ├── patients/               # Patient data samples
│   └── training/               # Training datasets
├── models/                     # Trained ML models (.pkl)
│   ├── breathing_model.pkl
│   ├── emotion_model.pkl
│   ├── glucose_model.pkl
│   ├── heartbeat_model.pkl
│   ├── posture_model.pkl
│   └── speech_model.pkl
├── recommendations/
│   └── engine.py               # Recommendation & insights engine
├── training/
│   ├── train_all.py
│   ├── train_breathing.py
│   ├── train_emotion.py
│   ├── train_glucose.py
│   ├── train_heartbeat.py
│   ├── train_posture.py
│   └── train_speech.py
├── utils/
│   ├── generate_data.py        # Synthetic data generation
│   └── inference.py            # Model inference utilities
├── frontend/                   # React frontend
├── .gitignore
└── README.md
```

---

## Key Features

- Multi-modal AI/ML models for rehabilitation and health monitoring
- Real-time inference and anomaly detection
- Flask-based backend serving ML predictions via REST APIs
- Interactive React dashboard for visualization and analytics
- Modular training and inference pipelines

---

## AI/ML Models

- Heartbeat abnormality detection (ECG analysis)
- Blood glucose estimation and risk prediction
- Breathing irregularity and apnea detection
- Posture and movement analysis
- Speech pattern and fluency analysis
- Emotional state detection

---

## Tech Stack

- Language: Python
- Machine Learning: TensorFlow, PyTorch, Scikit-learn
- Backend: Flask, REST APIs
- Frontend: React
- Database: PostgreSQL
- Tools: Git, Docker

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- Node.js
- PostgreSQL

### Backend Setup
```bash
git clone https://github.com/lokaesshwar/RehabSense.git
cd RehabSense/backend
pip install -r requirements.txt
python app.py
```

### Model Training
```bash
cd RehabSense/training
python train_all.py
```

### Frontend Setup
```bash
cd RehabSense/frontend
npm install
npm start
```

---

## How It Works

1. Training scripts generate and train ML models
2. Trained models are stored in the models directory
3. Flask backend loads models and exposes inference APIs
4. Recommendation engine generates insights from predictions
5. React frontend visualizes real-time and historical data

---

## Future Enhancements

- FastAPI-based asynchronous inference services
- Wearable and IoT sensor integration
- Explainable AI for clinical insights
- Mobile application support

---

## Disclaimer

This project is for educational and research purposes only and is not intended for medical diagnosis or treatment.

---

