Automated Road Extraction using SVM and XGBoost
📌 Project Overview

-> This project focuses on automated road extraction from satellite imagery using machine learning models (SVM and XGBoost). The goal is to classify pixels or image patches as road / non-road based on extracted features, enabling scalable and automated mapping.

-> The dataset used in this project is publicly available on Kaggle and is not included in this repository due to size and upload limitations.

🧠 Models Used
  
  -> Support Vector Machine (SVM)
  
  -> XGBoost (Extreme Gradient Boosting)

  -> U-Net

Both models are trained and evaluated to compare:
  -> Classification accuracy
  
  -> Precision, recall, and F1-score
  
  -> Generalization performance on unseen data

📊 Dataset

The dataset consists of satellite images and corresponding road masks/labels used for supervised learning.

🔗 Dataset Source (Kaggle)

Download the dataset from Kaggle using the link below:

👉 [Kaggle – Road Extraction Dataset]
(Replace this with the exact Kaggle dataset link you used)

⚠️ Note:
The dataset is not included in this repository because Kaggle datasets are large and cannot be uploaded directly to GitHub.

⚙️ Feature Extraction

Features extracted from satellite images include:

-> Pixel intensity values
-> Texture features
-> Statistical features (mean, variance, etc.)
-> Spatial information (optional)

These features are used as inputs to both SVM and XGBoost models.

🚀 Model Training

-> SVM: Effective for high-dimensional feature spaces and clear margin separation
-> XGBoost: Handles non-linear patterns and feature interactions efficiently
-> Hyperparameter tuning is performed to optimize performance.

📈 Evaluation Metrics

The models are evaluated using:

-> Accuracy
-> Precision
-> Recall
-> F1-score
-> Confusion Matrix

Comparative analysis is provided to highlight strengths and weaknesses of each model.
🛠️ Requirements

Install dependencies using:

 -> pip install -r requirements.txt

Main libraries:

 -> Python 3.x
 -> NumPy
 -> Pandas
 -> OpenCV / PIL
 -> Scikit-learn
 -> XGBoost
 -> Matplotlib / Seaborn

📝 Notes

This project is intended for educational and research purposes

Can be extended using CNNs or deep learning segmentation models

Suitable for GIS, remote sensing, and computer vision applications

👤 Author

Puneeth Raj Y
Automated Road Extraction using Machine Learning
SVM | XGBoost | Satellite Image Analysis
