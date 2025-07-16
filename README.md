🩺 Diabetic Retinopathy Detection – APTOS 2019
Imagine being able to detect blindness before it happens.

Diabetic Retinopathy (DR) is the leading cause of blindness in working-aged adults worldwide. In India, Aravind Eye Hospital aims to prevent this irreversible disease among people in rural regions by automating early detection. This project supports that mission using machine learning to identify the severity of DR from retinal fundus images.

Using thousands of images collected in underserved areas, this solution leverages both custom CNN models and transfer learning (ResNetV2-50) to accurately classify the stage of DR — from no disease to severe proliferative stages.

🌍 Real-World Motivation
Currently, Aravind Eye Hospital sends technicians to rural areas to capture retinal images. These are then manually reviewed by trained ophthalmologists — a slow, costly, and capacity-limited process.

By automating image screening through deep learning, this system will:

Speed up diagnosis

Help scale rural outreach programs

Prevent lifelong blindness

Provide a scalable model for other diseases (e.g., glaucoma, macular degeneration)

This project aligns with the goals of the 4th Asia Pacific Tele-Ophthalmology Society (APTOS) Symposium, helping to develop deployable, accurate AI systems in real-world ophthalmology.

🗂️ Dataset Description
📦 Source: APTOS 2019 Blindness Detection

🖼️ Data: Fundus images under varied conditions and equipment

🧠 Labels: DR severity (0–4) assigned by certified clinicians:

0: No DR

1: Mild

2: Moderate

3: Severe

4: Proliferative DR

📌 Real-World Challenges:
The data includes artifacts, blurring, over/underexposure, and camera variation. Models must generalize despite these variations.

🔍 Problem Statement
Your model should predict the probability (0.0–1.0) that an image contains malignant lesions. Though the training labels are multiclass (0–4), the evaluation task simplifies the classification into a binary probability of DR presence.

🚀 Solution Overview
🔹 Data Preprocessing
Downloaded with kagglehub

Image resizing, normalization

Augmentation (zoom, flip, brightness) using ImageDataGenerator

Train/test split with stratification

🔹 Models Used
🧱 Custom CNN
Convolutional layers with pooling, dropout, and batch normalization

Trained with Cosine Annealing for dynamic learning rate adaptation

Test Accuracy: ~75%

🏗️ ResNetV2-50 (Transfer Learning)
Pre-trained on ImageNet

Top layers replaced with custom dense layers

Fine-tuned on fundus images

Test Accuracy: ~80%

🔁 Learning Rate Strategy
✅ Cosine Annealing is used for the CNN to gradually reduce the learning rate and avoid local minima.

Implemented via tf.keras.callbacks.LearningRateScheduler.

📊 Model Performance Summary
Model	Epochs	Learning Rate Strategy	Test Accuracy
Custom CNN	10	Cosine Annealing	~75%
ResNetV2-50 (TL)	10	Reduced LR	~80%

🧰 Tech Stack
Python

TensorFlow / Keras

OpenCV

Scikit-learn

Seaborn, Matplotlib

kagglehub for dataset access

🗃️ Files Structure
train.csv – Training labels

test.csv – Test image IDs

train.zip – Training images

test.zip – Test images

Notebook – Full preprocessing, modeling, and evaluation code

