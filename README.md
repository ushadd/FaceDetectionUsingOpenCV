# FaceDetectionUsingOpenCV

An intelligent real-time face detection system built using OpenCV, DeepFace, and Face Recognition.
This project captures live video from your webcam, detects human faces, and identifies emotion and gender for each face in real time.

🚀 Features

👤 Real-time Face Detection using face_recognition

😊 Emotion Recognition (Happy, Sad, Angry, Neutral, etc.) with DeepFace

🚻 Gender Classification using a pre-trained Caffe model

🎥 Live Video Capture from webcam

🧠 Deep Learning-powered predictions with OpenCV DNN

🧩 Tech Stack
Component	Technology
Language	Python
Libraries	OpenCV, DeepFace, NumPy, face_recognition
Models Used	gender_net.caffemodel, gender_deploy.prototxt
Frameworks	OpenCV DNN, DeepFace backend
🧠 How It Works

Webcam Capture

The system continuously reads frames from your webcam using OpenCV.

Face Detection

Faces are detected using face_recognition.face_locations().

Emotion Recognition

The cropped face image is passed to DeepFace to detect the dominant emotion.

Gender Prediction

The OpenCV DNN module loads pre-trained gender detection models to classify the face as Male or Female.

Display

A bounding box and label (Gender + Emotion) are displayed above each detected face.

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/Face-Detection-using-OpenCV.git
cd Face-Detection-using-OpenCV

2️⃣ Install required dependencies
pip install opencv-python deepface face_recognition numpy

3️⃣ Download pre-trained models

Download these files and update their paths in the script:

gender_deploy.prototxt

gender_net.caffemodel

4️⃣ Run the project
python faceDetection.py


Press Q to exit the window.

📁 Project Structure
Face-Detection-using-OpenCV/
│
├── faceDetection.py             # Main program
├── gender_deploy.prototxt       # Model architecture
├── gender_net.caffemodel        # Pre-trained gender model
└── README.md                    # Project documentation

📸 Example Output

When you run the script, the webcam window will display bounding boxes around detected faces labeled with gender and emotion, for example:

Male, Happy
Female, Neutral

🧑‍💻 Author

Usha D D
🎓 BCA Fresher | AI & Computer Vision Enthusiast
🔗 [GitHub](https://github.com/ushadd)
 • [LinkedIn](https://www.linkedin.com/in/usha-d-d-261a46363)

🌟 Future Enhancements

🧒 Add Age Estimation model

🧍 Detect multiple people in frame with better accuracy

🎙️ Add voice feedback for predictions

📊 Integrate with dashboard for emotion statistics
