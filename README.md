# DL-Image-Calorie-Prediction

## Overview

This repository contains code for predicting the calorie content of food items based on their images. It performs the finetuning of the ResNet model.
Here, my model achieves an accuracy of 98% on the test dataset.
The best model can be found [here](https://drive.google.com/file/d/1hFTxJaXXGyia_iT0fL8V9mLAanYWCiAY/view?usp=sharing)


## The model: ResNet
![Original-ResNet-18-Architecture](https://github.com/dwipddalal/DL-Image-calorie-prediction/assets/91228207/e21e6b70-d0d3-4d51-92e1-7c5182cb3c23)


The ResNet (Residual Network) architecture revolutionized the field of deep learning by introducing the concept of residual connections, effectively solving the vanishing gradient problem in deep networks. Unlike traditional architectures that stack layers in a sequential manner, ResNet incorporates "skip connections" that bypass one or more layers. These connections allow the gradient to backpropagate more effectively through the network, enabling the training of much deeper models. The architecture consists of multiple "residual blocks," each containing convolutional layers followed by batch normalization and ReLU activation. The output from these layers is added to the input, forming the "residual" component. This additive nature of residual blocks allows the network to learn identity functions easily, providing a path for gradients during backpropagation. ResNet has been a foundational architecture for various tasks, setting new performance benchmarks and serving as a backbone for many state-of-the-art models.

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

- The script consist of train/test and also the inference module.

### Deployment
- Download the best_model from the link given above.
- Run the Flask app: python app.py

- /deployment
  - /static
    - index.html
    - script.js
  - app.py
  - best_model.pth




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






