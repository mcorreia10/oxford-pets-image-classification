# Oxford-IIIT Pet Breed Classification Project

## Project Overview

This project focuses on image classification using the **Oxford-IIIT Pet Dataset**, a dataset containing images of cats and dogs from 37 different breeds.

The main objective is to build, train, evaluate, and compare different deep learning models for pet breed classification. The project starts with a custom baseline CNN and then explores transfer learning approaches using more advanced architectures.

The final output includes:

- Exploratory Data Analysis (EDA)
- A custom CNN baseline model
- Transfer learning models
- Model comparison
- A Streamlit app for image inference

---

## Problem Statement

The goal is to classify an input image of a pet into one of the 37 available breed classes.

This is a fine-grained image classification problem because several breeds are visually similar. The model must learn differences in features such as fur texture, face shape, body structure, color patterns, and other breed-specific characteristics.

---

## Dataset

The project uses the **Oxford-IIIT Pet Dataset**, which contains images of cats and dogs across 37 classes.

Each class represents a specific breed.

The dataset is approximately balanced, with a similar number of images per class. This makes accuracy a reasonable metric for model comparison.

---

## Exploratory Data Analysis

The EDA section includes:

### Class Distribution

The class distribution analysis shows the number of images available for each breed. This helps verify whether the dataset is balanced or biased toward specific classes.

### Sample Images

Random samples from the dataset are displayed to inspect:

- image quality;
- background variation;
- different animal poses;
- lighting conditions;
- visual similarity between breeds.

This step helps justify the use of data augmentation and more robust model architectures.

### Pixel Statistics

Pixel statistics were also explored to understand the RGB channel distribution of the dataset and support preprocessing decisions such as image normalization.

---

## Models

The project compares several model families.

### 1. Baseline CNN

A custom CNN was built from scratch to establish a reference performance.

The baseline architecture includes:

- convolutional blocks;
- batch normalization;
- ReLU activations;
- max pooling;
- adaptive average pooling;
- dropout;
- fully connected classification head.

This model is trained from scratch and does not use pretrained weights.

The baseline is important because it provides a comparison point against transfer learning models.

### 2. ResNet

A pretrained ResNet model is used as a transfer learning approach.

The final classification layer is replaced to match the 37 pet breed classes. The model benefits from visual features learned previously on a large-scale image dataset.

### 3. ConvNeXt

ConvNeXt is used as a modern convolutional architecture. It combines ideas from traditional CNNs and more recent deep learning design improvements.

This model is expected to perform strongly due to its pretrained feature extractor and improved architecture.

### 4. Vision Transformer

A Vision Transformer model is also included for comparison.

Unlike CNN-based models, ViT processes images as patches and uses transformer-based attention mechanisms to learn visual representations.

---

## Training Strategy

The project uses different training strategies depending on the model type.

For the baseline CNN:

- the model is trained from scratch;
- image augmentation is applied during training;
- validation accuracy is monitored;
- the best model checkpoint is saved.

For transfer learning models:

- pretrained weights are used;
- the final classifier is adapted to 37 classes;
- selected layers may be fine-tuned;
- the best performing checkpoint is saved.

---

## Evaluation Metrics

The models are evaluated using:

- training loss;
- validation loss;
- training accuracy;
- validation accuracy;
- test accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix.

The main comparison metric is **test accuracy**, because it evaluates performance on unseen data.

Validation accuracy is used during training to select the best model checkpoint.

---

## Model Comparison

The final comparison includes:

- baseline CNN performance;
- ResNet performance;
- ConvNeXt performance;
- Vision Transformer performance.

The comparison is used to understand:

- how much transfer learning improves performance;
- whether modern pretrained models outperform the custom baseline;
- how well each model generalizes to unseen images;
- the gap between validation and test accuracy.

---

## Streamlit App

A Streamlit app was developed to run inference on uploaded pet images.

The app allows the user to:

- upload an image;
- select a trained model;
- run prediction;
- view the predicted breed;
- view the confidence score;
- inspect the top-3 predictions.

At the current stage, the app supports the baseline CNN. Additional trained models such as ConvNeXt, ResNet, and ViT can be added by placing their `.pth` files in the app model directory and updating the model configuration.

---

## How to Run the App

Go to the app folder:

```bash
cd 05_app
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open locally in the browser, usually at:

```text
http://localhost:8501
```

---

## Main Findings

The baseline CNN provides a useful reference point but has limitations when dealing with fine-grained breed classification.

Transfer learning models are expected to perform significantly better because they use pretrained visual representations learned from large-scale datasets.

The app demonstrates how trained deep learning models can be integrated into a simple user-facing interface for real-time inference.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- PIL / Pillow
- Jupyter Notebook / Google Colab

