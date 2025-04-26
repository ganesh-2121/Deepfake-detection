🔍 Deepfake Detection Using CNN
This project is a deep learning-based solution for detecting real vs. fake (deepfake) facial images using a Convolutional Neural Network (CNN). It uses a lightweight architecture trained on a balanced dataset of real and AI-generated (fake) face images. The solution includes a web-based interface built with Streamlit that allows users to upload an image and get an instant prediction.

📌 Table of Contents

About the Project

Features

Tech Stack

Dataset

Model Architecture

Usage

Results

Limitations & Future Scope

 About the Project

Deepfakes are synthetic media in which a person’s likeness is replaced with someone else’s using artificial intelligence. These can be highly deceptive and pose a threat to digital integrity. This project focuses on detecting such images using CNNs and showcasing results interactively via a web app.

 Features

Upload any face image and detect if it is Real or Fake

CNN-based binary image classification

Trained on 80,000 facial images (40k real, 40k fake)

Real-time predictions with probability score

Web UI built using Streamlit

Model Performance Summary with accuracy/loss

⚙️ Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy

Streamlit (for UI)

Matplotlib (for model visualization)

📂 Dataset

We used the 140K Real and Fake Face Dataset, which consists of: 70,000 Real Images collected from Flickr and real photo datasets.

70,000 Fake Images generated using StyleGAN and other AI generators.

Split used:

Training Set: 40,000 Real + 40,000 Fake

Validation Set: 10,000 Real + 10,000 Fake


🧠 Model Architecture

Input size: 96x96x3

Convolutional layers with ReLU activation

MaxPooling layers

Dropout for regularization

Flatten + Dense layers

Sigmoid activation for binary output



🚀 Usage

1.Launch the app.

2.Upload a .jpg, .jpeg, or .png face image.

3.The model will output whether the image is REAL or FAKE, along with confidence.

4.A brief explanation is shown based on the result.

	
📊 Results

Validation Accuracy: 93.98%

Validation Loss: 0.1577

CNN was able to distinguish between deepfakes and real images effectively.

Good generalization with unseen images from external sources.

🚧 Limitations & Future Scope

Limitations:

Focused on still images only, not videos.

May fail on highly realistic fake images with minimal artifacts.

Does not support batch image analysis.

Future Scope:

Extend to deepfake video detection.

Use transformer-based or attention mechanisms.

Add heatmap visualizations for feature attention.

Enable API endpoints for broader integrations.
