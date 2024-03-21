import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
from torchvision import models

# Web App Title
st.markdown('''
# **A Weather Classification App**

---
''')

# Define your CNN class
def linear_block(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.ReLU(),
        nn.Dropout(0.1)
    )

def conv_block(input_channels, output_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(input_channels, output_channels, (5, 5), padding=1),
        torch.nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.2)
    )

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            conv_block(3, 5),
            torch.nn.Flatten(),
            linear_block(49005, 1024),
            linear_block(1024, 64),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        out = self.main(x)
        return out

# Define your data transformations
data_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

def load_model():
    # Define the model architecture
    densenet = models.densenet121(pretrained=False)
    new_layers = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 4)
    )
    densenet.classifier = new_layers

    # Load the state dictionary
    densenet.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    densenet.eval()
    return densenet

model = load_model()

class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
classes = {i: class_name for i, class_name in enumerate(class_names)}


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')  # Convert image to RGB
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        with st.spinner('Predicting...'):
            # Preprocess the image
            img = data_transforms(image).unsqueeze(0)
            
            # Make predictions
            with torch.no_grad():
                img = img.to('cpu')
                y_hat = model(img)
                _, predicted = torch.max(y_hat, 1)
            
            # Calculate the probability
            probabilities = torch.nn.functional.softmax(y_hat, dim=1)
            probability = probabilities[0][predicted.item()]

            # Print the prediction and probability
            st.write(f'Predicted Class: {class_names[predicted.item()]}, Probability: {probability}')
            
            # If the model is unsure
            if probability < 0.95:
                st.write('The model is unsure about this prediction.')

