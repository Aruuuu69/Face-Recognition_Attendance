import cv2
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Paths
DATASET_DIR = 'dataset'
MODEL_DIR = 'trained_model'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'face_recognition_knn_model.pkl')
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Prepare Data
X = []  # Features (Flattened face images)
y = []  # Labels (Person names)

print("[INFO] Preparing dataset...")

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path)

        if img is None:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y_, w, h) in faces:
            face = gray[y_:y_+h, x:x+w]

            # Resize to a standard size (say 50x50)
            face_resized = cv2.resize(face, (50, 50))

            # Flatten the 2D face image into 1D feature vector
            face_flattened = face_resized.flatten()

            X.append(face_flattened)
            y.append(person_name)

print(f"[INFO] Dataset prepared with {len(X)} face samples.")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train KNN model
print("[INFO] Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=5)  # you can adjust k
knn.fit(X, y)

# Save the model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(knn, f)

print(f"[INFO] Model trained and saved at {MODEL_SAVE_PATH}")
