# LIVE-VIDEO-FACE-RECOGNITION-ATTENDANCE-SYSTEM
The project is a real-time face recognition attendance system using MTCNN, FaceNet, and KNN. It captures faces from live video, marks attendance in Firebase, and displays data on a secure web dashboard. It ensures accurate, contactless, and automated attendance tracking using AI and web technologies.
Face Recognition Attendance System  Project Documentation

Project Title:
Face Recognition-Based Real-Time Attendance System using FaceNet and Firebase

Objective:
To automate the attendance process using real-time face recognition with a webcam. This project captures faces, recognizes individuals using FaceNet embeddings and MTCNN detection, and marks attendance in a Firebase Realtime Database.

Key Technologies Used:
•	Programming Language: Python 3.x
•	Face Detection: MTCNN (Multi-task Cascaded Convolutional Networks)
•	Face Embedding: FaceNet (keras-facenet wrapper)
•	Classification Model: K-Nearest Neighbors (KNN)
•	Database: Firebase Realtime Database
•	GUI/Display: OpenCV
•	Data Visualization: matplotlib, seaborn
•	Audio Notification: playsound module

Required Python Packages:
Ensure the following packages are installed. You can use pip install package_name to install each:
pip install opencv-python
pip install numpy
pip install mtcnn
pip install keras-facenet
pip install firebase-admin
pip install scikit-learn
pip install playsound
pip install matplotlib
pip install seaborn

Software Requirements:
•	Python 3.6+
•	Any Python IDE or Text Editor (e.g., PyCharm, VS Code, Jupyter Notebook)
•	Google Chrome or Web Browser (to check Firebase DB)
•	Internet Connection (for Firebase communication)

Hardware Requirements:
•	Webcam (built-in or external)
•	System with at least 4GB RAM

File/Folder Structure:
project_directory/
|— face_recognizer.pkl            # Trained KNN model
|— snapshot/                      # Saved snapshots of recognized faces
|— ding.wav                      # Sound file played on successful recognition
|— facenetDB/                    # Directory containing training images
|   |— 211FA05146.jpg             # Naming convention: rollnumber.jpg
|— face_attendance.py            # Main Python script (your provided code)
|— firebase-adminsdk.json        # Firebase credentials JSON file

How It Works:
1.	Face Detection: Detects face in real-time using MTCNN.
2.	Preprocessing: Lighting normalization using CLAHE.
3.	Embedding: Converts face to 128-d vector using FaceNet.
4.	Classification: Predicts identity using a trained KNN model.
5.	Attendance Marking: Pushes data to Firebase with timestamp.
6.	Snapshot Saving: Saves the face image with timestamp locally.
7.	Audio Alert: Plays a sound when a face is recognized.

How to Train Model:
•	The model is trained automatically if face_recognizer.pkl is not found.
•	Images should be stored in facenetDB/ with filename as the roll number.
•	Each image undergoes augmentation to generate 50 variations for training.

Firebase Setup:
1.	Go to https://console.firebase.google.com
2.	Create a new project and enable Realtime Database.
3.	Download the admin SDK JSON file and place it in your project directory.
4.	Replace the credential path and database URL in the script.

How to Run:
python face_attendance.py
•	Press 'x' to stop webcam and end session.

Output:
•	Live webcam with name label on recognized faces.
•	Attendance entry created in Firebase.
•	Local snapshot of the recognized face.

Notes for Future Students:
•	You can add new students by placing their labeled images in facenetDB/ and retraining.
•	Use clear frontal images for better accuracy.
•	Firebase email/password auth is not required for Realtime DB.

Troubleshooting:
•	No camera detected: Check webcam permissions.
•	Model not accurate: Ensure dataset images are high quality and consistent.
•	Firebase error: Recheck JSON path and DB URL.

# Run
process_frame()
step-by-step execution guide for your Face Recognition Attendance System Python project, covering setup, training, and real-time execution:
1. Setup and Initializations
a. Import Required Libraries
•	Libraries include OpenCV, NumPy, MTCNN, Keras FaceNet, Firebase Admin, Scikit-learn, etc.
•	These are used for:
o	Face detection (MTCNN)
o	Face embedding (FaceNet)
o	Classification (KNN)
o	Attendance logging (Firebase)
o	Snapshot capture, GUI display, and sound alert
b. Initialize Models
detector = MTCNN()
embedder = FaceNet()
•	MTCNN: Detects face bounding boxes
•	FaceNet: Generates 128D face embeddings
2. Configure Paths and Firebase
•	Define paths to:
o	Dataset folder (DATASET_PATH)
o	Trained model output file (face_recognizer.pkl)
o	Attendance sound file (ding.wav)
o	Snapshot saving directory (snapshot)
•	Initialize Firebase using your .json credentials to access the realtime database for marking attendance.
3. Face Preprocessing and Augmentation
a. Lighting Normalization
•	Applies CLAHE to enhance local contrast.
b. Image Augmentation (50 variations per face)
•	Random:
o	Flip
o	Brightness/contrast
o	Rotation
o	Gaussian noise
o	Scaling (zoom in/out)
This increases dataset robustness during training.
4. Get and Normalize Embeddings
•	Extract 128D vector for each face image using FaceNet
•	Normalize embeddings to unit length for distance comparison
5. Train the Model (Only if Not Already Trained)
If face_recognizer.pkl does not exist:
train_model()
Steps:
1.	Read each image in the dataset folder.
2.	Detect the face → apply augmentation → extract embeddings.
3.	Fit KNN classifier using Euclidean distance.
4.	Save the trained model to disk.
5.	Show performance using accuracy, precision, confusion matrix, and classification report.
6.  Load Trained KNN Model
with open(MODEL_OUTPUT_PATH, "rb") as f:
    knn = pickle.load(f)
7. Real-Time Attendance Execution
a. Start Webcam
cap = cv2.VideoCapture(0)
b. For Each Frame:
1.	Detect faces using MTCNN
2.	For each detected face:
Crop & preprocess
Get embedding
Predict name using KNN
If matched and distance ≤ 0.8:
Mark attendance in Firebase
Play sound
Save snapshot
c. Exit Condition
Press 'x' to exit the camera loop.
d. Mark Absentees
Any name from registered_names not seen is marked Absent in Firebase.

 8. Snapshot Saving
Captures and saves snapshot of detected faces in the snapshot directory with a timestamped filename.
9. Audio Notification
When a known face is recognized and attendance is marked, it plays a beep sound using playsound().

_**Expected Output**_
A live webcam window showing bounding boxes and predicted names.
Firebase Realtime Database updated with:
Name
Time
Status (Present/Absent)
Snapshots saved locally
Sound plays on successful attendance mark
Print logs in the terminal
Typical Flow of Use
1.	Place face images (named as roll numbers) in facenetDB/
2.	Run the script → it trains if model is absent
3.	Launches webcam → detects & recognizes faces
4.	Marks attendance in Firebase and saves snapshots
5.	Ends when user presses 'x'
