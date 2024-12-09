# Function to extract face embeddings using custom CNN
def extract_face_embeddings(frame, face_coordinates):
    x, y, w, h = face_coordinates
    face = frame[y:y+h, x:x+w]
    face_tensor = transform(face).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        embedding = model(face_tensor)
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
    cv2.imshow('Face Recognition with YOLO and Custom CNN', frame)

    # Quit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
