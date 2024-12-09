#Pytorch and Yolo PreTrained

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from sklearn.neighbors import KNeighborsClassifier
import os

# Load pre-trained YOLO model (you'll need to download the weights and config)
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load pre-trained ResNet model for face recognition (you can use other models like VGG16 or FaceNet)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model = models.resnet18(pretrained=True).to(device)
resnet_model.eval()  # Set the model to evaluation mode

# Load previously trained KNN model for face recognition
knn = KNeighborsClassifier(n_neighbors=3)

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to detect faces using YOLO
def detect_faces_yolo(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    faces = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # Class 0 is for 'person' in YOLO
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                faces.append((center_x, center_y, w, h))
    return faces

# Function to extract face embeddings using ResNet
def extract_face_embeddings(frame, face_coordinates):
    x, y, w, h = face_coordinates
    face = frame[y:y+h, x:x+w]
    face_tensor = transform(face).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        embedding = resnet_model(face_tensor)
    return embedding.cpu().numpy().flatten()  # Convert to numpy array and flatten

# Function to recognize face using KNN classifier
def recognize_face(embedding):
    predicted_class = knn.predict([embedding])
    return predicted_class

# Function to save unknown faces
def save_unknown_face(frame, face_coordinates):
    x, y, w, h = face_coordinates
    unknown_face = frame[y:y+h, x:x+w]
    filename = f"unknown_faces/face_{len(os.listdir('unknown_faces'))}.jpg"
    cv2.imwrite(filename, unknown_face)

# Initialize webcam and start recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detect_faces_yolo(frame)
    for face_coordinates in faces:
        embedding = extract_face_embeddings(frame, face_coordinates)
        prediction = recognize_face(embedding)
        
        # If the face is unrecognized, save it for retraining
        if prediction == 'Unknown':  # You can set a threshold for "unknown"
            save_unknown_face(frame, face_coordinates)
        
        # Draw rectangle around recognized or unknown face
        x, y, w, h = face_coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame with recognized face or "Unknown" label
    cv2.imshow('Face Recognition with YOLO and PyTorch', frame)

    # Quit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
