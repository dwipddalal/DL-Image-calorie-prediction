# DL-Image-Calorie-Prediction

## Overview

This repository contains code for predicting the calorie content of food items based on their images. It performs the finetuning of the ResNet model.
Here, my model achieves an accuracy of 98% on the test dataset.
The best model can be found [here](https://drive.google.com/file/d/1hFTxJaXXGyia_iT0fL8V9mLAanYWCiAY/view?usp=sharing)

## Structure

- `main.py`: Contains the core logic for training the deep learning model.
- [`deployment/app.py`](https://github.com/dwipddalal/DL-Image-calorie-prediction/blob/main/deployment/app.py): Flask application for serving the model.
- [`deployment/static/script.js`](https://github.com/dwipddalal/DL-Image-calorie-prediction/blob/main/deployment/static/script.js): JavaScript for handling image upload and displaying prediction results.

## Requirements

- Python 3.x
- PyTorch
- Flask
- PIL

## Usage

### Training the Model

1. Clone the repository:



