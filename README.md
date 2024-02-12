# Weather Image Classification Project

This project focuses on the classification of images depicting various weather conditions. The dataset used is the “Multi-class Weather Dataset for Image Classification” (Ajayi, Gbeminiyi, 2018, Mendeley Data, V1, doi: 10.17632/4drtyfjtfy.1).

## Overview
The project employs PyTorch to implement a Convolutional Neural Network (CNN) model for the classification task. The approach to the classification task is threefold:
1. Building a CNN from scratch.
2. Leveraging a pre-trained (DenseNet121) model as a feature extractor.
3. Leveraging a pre-trained (DenseNet121) model and fine-tuning it.
These models have achieved a satisfactory result, with 0.8 validation accuracy for the model built from scratch, and over 0.95 validation accuracy after leveraging a pre-trained model.

Details of the analysis can be found [here](https://github.com/billwan96/2024_02-Weather_Image_Classification/blob/main/analysis.ipynb)

## Dataset
The “Multi-class Weather Dataset for Image Classification” is a collection of images under various weather conditions. It’s a great resource for developing and testing machine learning models for weather classification tasks. For the most accurate information, please refer to the dataset documentation or the original source.

## Models
1. CNN from Scratch
A custom CNN model is built from scratch and trained on the weather images.

2. Pre-trained DenseNet121 as Feature Extractor
A pre-trained DenseNet121 model is used as a fixed feature extractor. The classifier part of the DenseNet model is replaced with a new sequential model, which is trained on the weather images.

3. Fine-tuned Pre-trained DenseNet121
A pre-trained DenseNet121 model is fine-tuned on the weather images. Most of the layers in the DenseNet model are frozen, and the classifier is replaced with a new one.

## References
1. Ajayi, Gbeminiyi (2018), “Multi-class Weather Dataset for Image Classification”, Mendeley Data, V1, doi: 10.17632/4drtyfjtfy.1.
2. PyTorch Documentation. Available at: https://pytorch.org/docs/stable/index.html
3. DenseNet Models. Available at: https://pytorch.org/hub/pytorch_vision_densenet/
