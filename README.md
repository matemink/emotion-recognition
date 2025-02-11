# 🎭 Face Emotion Recognition with Stable Diffusion & Random Forest

## 🚀 Overview
This project leverages **Stable Diffusion**, **Mediapipe**, and a **Random Forest classifier** to generate and classify facial emotions. The pipeline includes **image generation**, **facial landmark extraction**, **emotion classification**, and **real-time recognition** from a webcam feed.

## 📝 Description
The goal of this project is to generate realistic facial expressions and train a model for emotion recognition. By using **Stable Diffusion** for image generation and **Mediapipe** for feature extraction, we classify emotions using a **Random Forest model**. Additionally, the system can predict emotions in real-time using a webcam.

## ⚡ Features
- 🎨 **Image Generation** – Uses **Stable Diffusion** to create photorealistic facial expressions.  
- 📍 **Facial Landmark Extraction** – Detects key facial features using **Mediapipe**.  
- 🧠 **Emotion Classification** – Trains a **Random Forest model** to categorize emotions.  
- 🎥 **Real-Time Detection** – Recognizes emotions from a live **webcam feed**.  

## 📦 Dependencies
Ensure you have the following libraries installed:
- **PyTorch**  
- **Diffusers (Stable Diffusion)**  
- **OpenCV**  
- **Mediapipe**  
- **NumPy**  
- **Scikit-learn**  

## 📌 Installation
```bash
pip install torch diffusers opencv-python mediapipe numpy scikit-learn
