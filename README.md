# Facial Expression Recognition Across Contexts
This project focuses on facial expression recognition across various contexts, such as cultural variations and lighting conditions. The goal is to identify emotions in images using a CNN deep learning network built with TensorFlow.

## Features
Facial Emotion Detection: Analyze emotions in uploaded images.
Contextual Adaptability: Designed to work under different conditions like diverse cultural backgrounds and varying lighting.
Deep Learning Model: Uses a Convolutional Neural Network (CNN) for accurate emotion prediction.

## Workflow
Data Collection: Datasets of facial expressions captured under different contexts were collected and pre-processed.
Model Training: Separate models were trained for specific contexts, or a unified multi-context model was developed.
Evaluation: The model was tested for its ability to generalize across unseen contexts.

## How to Use
1.Clone the repository:
git clone https://github.com/Hariarul/Facial-Emotion-classifier

2.Install the required dependencies:
pip install -r requirements.txt

3.Run the application:
Face_Ex.py

4.Upload an image to detect emotions.

## Dependencies
streamlit==1.40.1

numpy==1.26.4

tensorflow==2.17.1

Pillow==10.4.0

## Results
The model demonstrates its capability to adapt to different contexts and accurately classify emotions such as happiness, sadness, anger, and surprise.

## Future Improvements
Include real-time emotion detection.

Expand datasets to cover more diverse contexts.

Optimize the model for faster inference.









