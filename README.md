# DL-Image-Calorie-Prediction

## Overview

This repository contains code for predicting the calorie content of food items based on their images. It performs the finetuning of the ResNet model.
Here, my model achieves an accuracy of 98% on the test dataset.
The best model can be found [here](https://drive.google.com/file/d/1hFTxJaXXGyia_iT0fL8V9mLAanYWCiAY/view?usp=sharing)


## The model: ResNet
![Original-ResNet-18-Architecture](https://github.com/dwipddalal/DL-Image-calorie-prediction/assets/91228207/e21e6b70-d0d3-4d51-92e1-7c5182cb3c23)


## Structure

- `main.py`: Contains the core logic for training the deep learning model.
- [`deployment/app.py`](https://github.com/dwipddalal/DL-Image-calorie-prediction/blob/main/deployment/app.py): Flask application for serving the model.
- [`deployment/static/script.js`](https://github.com/dwipddalal/DL-Image-calorie-prediction/blob/main/deployment/static/script.js): JavaScript for handling image upload and displaying prediction results.

## Requirements

- Python 3.x
- PyTorch
- Flask

## Usage

### Training the Model

1. Clone the repository: git clone https://github.com/dwipddalal/DL-Image-calorie-prediction.git
2.  Navigate to the repository folder: cd DL-Image-calorie-prediction
3.  Run the main script: python main.py

### Deployment
Run the Flask app: python app.py

![WhatsApp Image 2023-09-28 at 18 40 25](https://github.com/dwipddalal/DL-Image-calorie-prediction/assets/91228207/19834057-f046-46ec-9106-df40dd997873)


Open your browser and go to `http://localhost:5000/` to interact with the model.

## Features
- **Data Loading**: Custom DataLoader for loading food images.
- **Model**: Uses a pre-trained ResNet model.
- **Training and Testing**: Includes code for training and testing the model.
- **Deployment**: Flask application for serving the model.
- **Frontend**: Simple HTML and JavaScript for user interaction.

## Contributing
Feel free to open issues or PRs if you find any problems or have suggestions for improvements.






