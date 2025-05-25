import cv2
import os

# Path to save captured faces
DATASET_DIR = 'dataset'
# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to create folder for a new person
def create_person_folder(name):
    person_folder = os.path.join(DATASET_DIR, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_faces():
    name = input("Enter the name of the person: ").strip()
    if not name:
        print("Name cannot be empty!")
        return

    person_folder = create_person_folder(name)

    cap = cv2.VideoCapture(0)

    print("\n[INFO] Starting face capture. Look at the camera and wait...")
    count = 0  # Number of images captured

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale (for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop the face region
            face_img = frame[y:y+h, x:x+w]

            # Resize to a standard size (optional)
            face_img = cv2.resize(face_img, (200, 200))

            # Save the face image
            file_path = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)

            count += 1

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Show the frame
        cv2.imshow('Capturing Faces', frame)

        # Break if 'q' is pressed OR if 50 samples captured
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    print(f"\n[INFO] Successfully captured {count} images for {name}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()
