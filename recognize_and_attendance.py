import cv2
import numpy as np
import pickle
import os
import pandas as pd
from datetime import datetime

# Paths
MODEL_PATH = 'trained_model/face_recognition_knn_model.pkl'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
ATTENDANCE_DIR = 'attendance'

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Load the trained KNN model
with open(MODEL_PATH, 'rb') as f:
    knn = pickle.load(f)

# Function to mark attendance
def mark_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

    if not os.path.exists(ATTENDANCE_DIR):
        os.makedirs(ATTENDANCE_DIR)

    # If file doesn't exist, create it with headers
    if not os.path.isfile(attendance_file):
        df = pd.DataFrame(columns=['Name', 'Time'])
        df.to_csv(attendance_file, index=False)

    # Read existing attendance
    df = pd.read_csv(attendance_file)

    # Mark attendance only once per person
    if name not in df['Name'].values:
        new_entry = {'Name': name, 'Time': time_str}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        # Save the attendance file
        try:
            df.to_csv(attendance_file, index=False)
            print(f"[INFO] Marked attendance for {name} at {time_str}")
        except PermissionError:
            print(f"[ERROR] Permission denied while saving attendance file.")

# Function to recognize faces
def recognize_faces():
    cap = cv2.VideoCapture(0)

    print("\n[INFO] Starting real-time face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        name = "Unknown"  # Default name if no face is detected

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (50, 50))  # Same size as training
            face_flattened = face_resized.flatten().reshape(1, -1)  # Reshape for prediction

            # Predict
            prediction = knn.predict(face_flattened)
            name = prediction[0]  # Assign recognized name

            # Draw rectangle and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Show welcome message when recognized
            if name != "Unknown":
                cv2.putText(frame, f"Welcome, {name}!", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 255), 2, cv2.LINE_AA)

            # Mark attendance
            mark_attendance(name)

        # Save unknown face only if a face is detected
        if len(faces) > 0 and name == "Unknown":
            unknown_faces_folder = "unknown_faces"
            if not os.path.exists(unknown_faces_folder):
                os.makedirs(unknown_faces_folder)
            face_filename = os.path.join(unknown_faces_folder, f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(face_filename, frame)
            print(f"[INFO] Saved unknown face at {face_filename}")

        # Show the frame
        cv2.imshow('Face Recognition Attendance System', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n[INFO] Exiting and saving attendance.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
