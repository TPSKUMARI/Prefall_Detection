# Prefall Detection

A computer-vision system for detecting abnormal gait patterns and fall risk in real time using a webcam. It uses MediaPipe Pose to track body landmarks, computes joint angles (knees, hips) and step metrics, and flags anomalies that may indicate an unsteady or pre-fall gait pattern.

## Files

- **prefall_detection_1.py** — Real-time gait anomaly detector. Uses MediaPipe Pose to track joint angles and step height, with a calibration phase to learn a person's normal gait, then flags statistically abnormal movement using persistence-based anomaly detection. Includes optional voice/audio alerts.
- **test_realtime_ano_2.py** — Extended gait anomaly detection system with a Tkinter GUI, additional signal processing (Savitzky-Golay filtering, z-score analysis), and optional machine learning based anomaly detection (Isolation Forest, One-Class SVM, and a PyTorch model).

## Requirements

- Python 3.x
- opencv-python
- mediapipe
- numpy
- pandas
- scipy
- pyttsx3 (optional, for voice feedback)
- torch (optional, for deep learning anomaly detection)
- scikit-learn (optional, for Isolation Forest / One-Class SVM)

Install core dependencies:

```bash
pip install opencv-python mediapipe numpy pandas scipy pyttsx3
```

Optional extras for the ML-based detector:

```bash
pip install torch scikit-learn
```

## Usage

Run the basic real-time detector:

```bash
python prefall_detection_1.py
```

Run the GUI-based detector with extended analysis:

```bash
python test_realtime_ano_2.py
```

Follow the on-screen calibration instructions to let the system learn a normal gait baseline before anomaly detection begins.
