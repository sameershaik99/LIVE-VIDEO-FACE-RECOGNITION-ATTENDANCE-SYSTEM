import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
import os
import firebase_admin
from firebase_admin import credentials, db
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
from playsound import playsound  # ✅ Sound module

# Initialize Face Detector & Embedder
try:
    detector = MTCNN()
    embedder = FaceNet()
except Exception as e:
    print("Error initializing MTCNN or FaceNet:", e)
    exit()

# Paths
DATASET_PATH = r"D:\mtcnn\facenetDB"
MODEL_OUTPUT_PATH = "face_recognizer.pkl"
SNAPSHOT_DIR = r"D:\mtcnn\snapshot"
SOUND_PATH = r"D:\mtcnn\ding.wav"  # ✅ Your sound file
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Firebase Setup
cred_path = r"D:\mtcnn\face-recognition-attenda-653fb-firebase-adminsdk-fbsvc-8207c166c6.json"
if not os.path.exists(cred_path):
    print("Error: Firebase credential file not found.")
    exit()

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-attenda-653fb-default-rtdb.firebaseio.com/"
})
attendance_ref = db.reference("Attendance")

# Registered Names
registered_names = ["211FA05146", "211FA05189", "211FA05283", "211FA05308", "211FA05309", "221LA05006","211FA05204","211FA05252","211FA05241","211FA05183","GSR"]

# Normalize lighting using CLAHE
def normalize_lighting(face):
    lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return normalized

# Augment image
def augment_image(img):
    aug_imgs = [img]
    aug_imgs.append(cv2.flip(img, 1))
    aug_imgs.append(cv2.convertScaleAbs(img, alpha=1.2, beta=30))
    aug_imgs.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-30))
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    aug_imgs.append(cv2.add(img, noise))
    return aug_imgs

# Get face embedding
def get_embedding(face_img):
    try:
        face = normalize_lighting(face_img)
        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)
        embedding = embedder.embeddings(face)
        return embedding[0]
    except:
        return None

# Normalize embedding
def normalize(embedding):
    return embedding / np.linalg.norm(embedding)

# Train Model
def train_model():
    embeddings = []
    labels = []

    for file in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, file)
        if os.path.isfile(path):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(img_rgb)

            if detections:
                x, y, w, h = detections[0]['box']
                x, y = max(0, x), max(0, y)
                face = img_rgb[y:y+h, x:x+w]
                augmented_faces = augment_image(face)

                for aug_face in augmented_faces:
                    embedding = get_embedding(aug_face)
                    if embedding is not None:
                        label = os.path.splitext(file)[0]
                        embeddings.append(embedding)
                        labels.append(label)

    if embeddings:
        knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        knn.fit(embeddings, labels)
        with open(MODEL_OUTPUT_PATH, 'wb') as f:
            pickle.dump(knn, f)
        predictions = knn.predict(embeddings)
        acc = accuracy_score(labels, predictions)
        print(f"✅ Model trained and saved. Training Accuracy: {acc*100:.2f}%")
    else:
        print("❌ No embeddings generated. Check your dataset.")

# Load Model
if not os.path.exists(MODEL_OUTPUT_PATH):
    print("Model not found. Training a new one...")
    train_model()

with open(MODEL_OUTPUT_PATH, "rb") as f:
    knn = pickle.load(f)

# Mark Attendance
def mark_attendance(name, seen_names):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if name not in seen_names:
        attendance_ref.child(date_str).child(name).set({
            "name": name,
            "time": time_str,
            "status": "Present"
        })
        seen_names.add(name)
        print(f"✅ Attendance marked for {name} at {time_str}")
        playsound(SOUND_PATH)  # ✅ Play sound on recognition

# Mark Absentees
def mark_absentees(seen_names):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    for name in registered_names:
        if name not in seen_names:
            attendance_ref.child(date_str).child(name).set({
                "name": name,
                "time": time_str,
                "status": "Absent"
            })
            print(f"❌ Marked {name} as Absent at {time_str}")

# Real-Time Frame Processing
def process_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    seen_names = set()
    saved_names = set()

    print("📷 Press 'x' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

        for det in detections:
            x, y, w, h = det['box']
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]

            embedding = get_embedding(face)
            name = "Unknown"

            if embedding is not None:
                embedding = normalize(embedding)
                name = knn.predict([embedding])[0]
                distance = knn.kneighbors([embedding])[0][0][0]

                if distance > 0.8:
                    name = "Unknown"
                elif name in registered_names:
                    mark_attendance(name, seen_names)

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            # Save snapshot
            if name != "Unknown" and name not in saved_names:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                snapshot_path = os.path.join(SNAPSHOT_DIR, filename)
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(snapshot_path, face_bgr)
                saved_names.add(name)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    mark_absentees(seen_names)
    cap.release()
    cv2.destroyAllWindows()

# Run
process_frame()