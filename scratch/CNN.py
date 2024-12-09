import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

# Custom CNN for face embeddings
class CustomFaceNet(nn.Module):
    def __init__(self):
        super(CustomFaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)  # Assuming input images are 224x224
        self.fc2 = nn.Linear(1024, 128)  # Output embedding of size 128

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = CustomFaceNet()

# Load previously trained KNN model for face recognition
knn = KNeighborsClassifier(n_neighbors=3)

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
